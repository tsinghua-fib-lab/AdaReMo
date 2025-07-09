import math
import pickle, json
import time
from tqdm import tqdm
import torch

from khrylib.utils import *
from khrylib.utils.torch import *
from khrylib.rl.agents import AgentPPO
from khrylib.rl.core import estimate_advantages, LoggerRL
from torch.utils.tensorboard import SummaryWriter
from news.envs import NewsEnv
from news.models.model import create_RL_model, create_GNN1_model, create_GNN2_model, create_GNN3_model, ActorCritic
from news.models.baseline import RandomPolicy, GreedyPolicy, NullModel
from news.utils.tools import TrajBatchDisc
from news.utils.config import Config


def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) for x in y] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


class NewsExpansionAgent(AgentPPO):

    def __init__(self,
                 cfg: Config,
                 dtype: torch.dtype,
                 device: torch.device,
                 num_threads: int,
                 training: bool = True,
                 checkpoint: Union[int, Text] = 0,
                 restore_best_rewards: bool = True):
        self.cfg = cfg
        self.training = training
        self.device = device
        self.loss_iter = 0
        self.setup_logger(num_threads)
        self.setup_env()
        self.setup_model()
        self.setup_optimizer()
        if checkpoint != 0:
            self.start_iteration = self.load_checkpoint(
                checkpoint, restore_best_rewards)
        else:
            self.start_iteration = 0
        super().__init__(env=self.env,
                         dtype=dtype,
                         device=device,
                         logger_cls=LoggerRL,
                         traj_cls=TrajBatchDisc,
                         num_threads=num_threads,
                         policy_net=self.policy_net,
                         value_net=self.value_net,
                         optimizer=self.optimizer,
                         opt_num_epochs=cfg.num_optim_epoch,
                         gamma=cfg.gamma,
                         tau=cfg.tau,
                         clip_epsilon=cfg.clip_epsilon,
                         value_pred_coef=cfg.value_pred_coef,
                         entropy_coef=cfg.entropy_coef,
                         policy_grad_clip=[(self.policy_net.parameters(), 1),
                                           (self.value_net.parameters(), 1)],
                         mini_batch_size=cfg.mini_batch_size)

    def sample_worker(self, pid, queue, num_samples, mean_action):
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls(**self.logger_kwargs)

        while logger.num_steps < num_samples:
            state = self.env.reset()

            last_info = dict()
            episode_success = False
            logger_messages = []
            memory_messages = []
            for t in range(10000):
                state_var = tensorfy([state])
                use_mean_action = mean_action or torch.bernoulli(
                    torch.tensor([1 - self.noise_rate])).item()
                # action = self.policy_net.select_action(state_var, use_mean_action).numpy().squeeze(0)
                action = self.policy_net.select_action(
                    state_var, use_mean_action).numpy().squeeze(0)

                next_state, reward, done, info = self.env.step(
                    action, self.thread_loggers[pid])
                # cache logging
                logger_messages.append([reward, info])

                mask = 0 if done else 1
                exp = 1 - use_mean_action
                # cache memory
                memory_messages.append(
                    [state, action, mask, next_state, reward, exp])

                if done:
                    episode_success = (reward != self.env.FAILURE_REWARD) and (
                        reward != self.env.INTERMEDIATE_REWARD)
                    last_info = info
                    break
                state = next_state

            if episode_success:
                logger.start_episode(self.env)
                for var in range(len(logger_messages)):
                    logger.step(self.env, *logger_messages[var])
                    self.push_memory(memory, *memory_messages[var])
                logger.end_episode(last_info)
            #     self.thread_loggers[pid].info(
            #         'worker {} finished episode {}.'.format(
            #             pid, logger.num_episodes))
            # else:
            #     self.thread_loggers[pid].info(
            #         '[!]worker {} failed episode {}.'.format(
            #             pid, logger.num_episodes))

        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger

    def setup_env(self):
        self.env = env = NewsEnv(self.cfg)
        self.numerical_feature_size = env.get_numerical_feature_size()
        self.node_dim = env.get_node_dim()
        self.edge_dim = env.get_edge_dim()
        self.max_num_nodes = env.get_max_node_num()
        self.max_num_edges = env.get_max_edge_num()

        # with open(self.cfg.log_dir + '/network.json', 'w') as f:
        #     env_info_dict = self.env.get_env_info_dict()
        #     json.dump(env_info_dict, f, indent=4)

    def setup_logger(self, num_threads):
        cfg = self.cfg
        self.tb_logger = SummaryWriter(cfg.tb_dir) if self.training else None
        self.logger = create_logger(os.path.join(
            cfg.log_dir, f'log_{"train" if self.training else "eval"}.txt'),
                                    file_handle=True)
        self.reward_offset = 0.0
        self.best_rewards = -1000.0
        self.best_plans = []
        self.current_rewards = -1000.0
        self.current_plans = []
        self.save_best_flag = False
        cfg.log(self.logger, self.tb_logger)

        self.thread_loggers = []
        for i in range(num_threads):
            self.thread_loggers.append(
                create_logger(os.path.join(
                    cfg.log_dir,
                    f'log_{"train" if self.training else "eval"}_{i}.txt'),
                              file_handle=True))

    def setup_model(self):
        cfg = self.cfg
        if cfg.agent == 'rl':
            self.policy_net, self.value_net = create_RL_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)

        elif cfg.agent == 'rl-gnn1':
            self.policy_net, self.value_net = create_GNN1_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'rl-gnn2':
            self.policy_net, self.value_net = create_GNN2_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'rl-gnn3':
            self.policy_net, self.value_net = create_GNN3_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'greedy':
            self.policy_net = GreedyPolicy()
            self.value_net = NullModel()

        elif cfg.agent == 'random':
            self.policy_net = RandomPolicy()
            self.value_net = NullModel()
        else:
            raise NotImplementedError()

    def setup_optimizer(self):
        cfg = self.cfg
        if cfg.agent in ['rl','rl-gnn1','rl-gnn2','rl-gnn3']:
            self.optimizer = torch.optim.Adam(
                self.actor_critic_net.parameters(),
                lr=cfg.lr,
                eps=cfg.eps,
                weight_decay=cfg.weightdecay)
        else:
            self.optimizer = None

    def load_checkpoint(self, checkpoint, restore_best_rewards):
        cfg = self.cfg
        if isinstance(checkpoint, int):
            cp_path = '%s/iteration_%04d.p' % (cfg.model_dir, checkpoint)
        else:
            assert isinstance(checkpoint, str)
            cp_path = '%s/%s.p' % (cfg.model_dir, checkpoint)
        self.logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        self.actor_critic_net.load_state_dict(model_cp['actor_critic_dict'])
        self.loss_iter = model_cp['loss_iter']
        if restore_best_rewards:
            self.best_rewards = model_cp.get('best_rewards', self.best_rewards)
            self.best_plans = model_cp.get('best_plans', self.best_plans)
        self.current_rewards = model_cp.get('current_rewards',
                                            self.current_rewards)
        self.current_plans = model_cp.get('current_plans', self.current_plans)
        start_iteration = model_cp['iteration'] + 1
        return start_iteration

    def save_checkpoint(self, iteration):

        def save(cp_path):
            with to_cpu(self.policy_net, self.value_net):
                model_cp = {
                    'actor_critic_dict': self.actor_critic_net.state_dict(),
                    'loss_iter': self.loss_iter,
                    'best_rewards': self.best_rewards,
                    'best_plans': self.best_plans,
                    'current_rewards': self.current_rewards,
                    'current_plans': self.current_plans,
                    'iteration': iteration
                }
                pickle.dump(model_cp, open(cp_path, 'wb'))

        cfg = self.cfg

        if cfg.save_model_interval > 0 and (iteration +
                                            1) % cfg.save_model_interval == 0:
            self.tb_logger.flush()
            save('{}/iteration_{:04d}.p'.format(cfg.model_dir, iteration + 1))
        if self.save_best_flag:
            self.tb_logger.add_scalar('best_reward/best_reward',
                                      self.best_rewards, iteration)
            self.tb_logger.flush()
            self.logger.info(
                f'save best checkpoint with rewards {self.best_rewards:.2f}!')
            save('{}/best.p'.format(cfg.model_dir))
            save('{}/best_reward{:.2f}_iteration_{:04d}.p'.format(
                cfg.model_dir, self.best_rewards, iteration + 1))

    def save_plan(self, log_eval: LoggerRL) -> None:
        """
        Save the current plan to file.

        Args:
            log_eval: LoggerRL object.
        """
        cfg = self.cfg
        self.logger.info(f'save plan to file: {cfg.plan_dir}\plan.p')
        with open(f'{cfg.plan_dir}\plan.p', 'wb') as f:
            pickle.dump(log_eval.plans, f)

    def optimize(self, iteration):
        info = self.optimize_policy(iteration)
        self.log_optimize_policy(iteration, info)

    def optimize_policy(self, iteration):
        """generate multiple trajectories that reach the minimum batch_size"""
        t0 = time.time()
        num_samples = self.cfg.num_episodes_per_iteration * self.cfg.max_sequence_length
        batch, log = self.sample(num_samples)
        """update networks"""
        t1 = time.time()
        if num_samples > 1:
            self.update_params(batch, iteration)
        t2 = time.time()
        """evaluate policy"""
        log_eval = self.eval_agent(num_samples=1, mean_action=True,visualize=False)
        t3 = time.time()

        info = {
            'log': log,
            'log_eval': log_eval,
            'T_sample': t1 - t0,
            'T_update': t2 - t1,
            'T_eval': t3 - t2,
            'T_total': t3 - t0
        }
        return info

    def update_params(self, batch, iteration):
        t0 = time.time()
        to_train(*self.update_modules)
        states = batch.states
        actions = torch.from_numpy(batch.actions).to(self.dtype)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype)
        masks = torch.from_numpy(batch.masks).to(self.dtype)
        exps = torch.from_numpy(batch.exps).to(self.dtype)
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = []
                chunk = self.cfg.mini_batch_size
                for i in range(0, len(states), chunk):
                    states_i = tensorfy(states[i:min(i + chunk, len(states))],
                                        self.device)
                    values_i = self.value_net(self.trans_value(states_i))
                    values.append(values_i.cpu())
                values = torch.cat(values)
        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values,
                                                  self.gamma, self.tau)
        self.update_policy(states, actions, returns, advantages, exps,
                           iteration)

        return time.time() - t0

    def get_perm_batch_stage(self, states):
        inds = [[], []]
        for i, x in enumerate(states):
            stage = x[-1]
            inds[stage.argmax()].append(i)
        perm = np.array(inds[0] + inds[1])
        return perm, LongTensor(perm)

    def update_policy(self, states, actions, returns, advantages, exps,
                      iteration):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = self.cfg.mini_batch_size
                for i in range(0, len(states), chunk):
                    states_i = tensorfy(states[i:min(i + chunk, len(states))],
                                        self.device)
                    actions_i = actions[i:min(i + chunk, len(states))].to(
                        self.device)
                    fixed_log_probs_i, _ = self.policy_net.get_log_prob_entropy(
                        self.trans_policy(states_i), actions_i)
                    fixed_log_probs.append(fixed_log_probs_i.cpu())
                fixed_log_probs = torch.cat(fixed_log_probs)
        num_state = len(states)

        tb_logger = self.tb_logger
        total_loss = 0.0
        total_value_loss = 0.0
        total_surr_loss = 0.0
        total_entropy_loss = 0.0
        for epoch in range(self.opt_num_epochs):
            epoch_loss = 0.0
            epoch_value_loss = 0.0
            epoch_surr_loss = 0.0
            epoch_entropy_loss = 0.0

            perm_np = np.arange(num_state)
            np.random.shuffle(perm_np)
            perm = LongTensor(perm_np)

            states, actions, returns, advantages, fixed_log_probs, exps = \
                index_select_list(states, perm_np), actions[perm].clone(), returns[perm].clone(), \
                advantages[perm].clone(), fixed_log_probs[perm].clone(), exps[perm].clone()

            if self.cfg.agent_specs.get('batch_stage', False):
                perm_stage_np, perm_stage = self.get_perm_batch_stage(states)
                states, actions, returns, advantages, fixed_log_probs, exps = \
                    index_select_list(states, perm_stage_np), actions[perm_stage].clone(), \
                    returns[perm_stage].clone(), advantages[perm_stage].clone(), \
                    fixed_log_probs[perm_stage].clone(), exps[perm_stage].clone()

            optim_batch_num = int(math.floor(num_state / self.mini_batch_size))
            for i in range(optim_batch_num):
                ind = slice(i * self.mini_batch_size,
                            min((i + 1) * self.mini_batch_size, num_state))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], exps[ind]
                ind = exps_b.nonzero(as_tuple=False).squeeze(1)
                states_b = tensorfy(states_b, self.device)
                actions_b, advantages_b, returns_b, fixed_log_probs_b, ind = batch_to(
                    self.device, actions_b, advantages_b, returns_b,
                    fixed_log_probs_b, ind)
                value_loss = self.value_loss(states_b, returns_b)
                surr_loss, entropy_loss = self.ppo_entropy_loss(
                    states_b, actions_b, advantages_b, fixed_log_probs_b, ind)
                loss = surr_loss + self.value_pred_coef * value_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.clip_policy_grad()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_value_loss += value_loss.item()
                epoch_surr_loss += surr_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                tb_logger.add_scalar('loss/loss', loss.item(), self.loss_iter)
                tb_logger.add_scalar('loss/value_loss', value_loss.item(),
                                     self.loss_iter)
                tb_logger.add_scalar('loss/surr_loss', surr_loss.item(),
                                     self.loss_iter)
                tb_logger.add_scalar('loss/entropy_loss', entropy_loss.item(),
                                     self.loss_iter)
                self.loss_iter += 1

            total_loss += epoch_loss
            total_value_loss += epoch_value_loss
            total_surr_loss += epoch_surr_loss
            total_entropy_loss += epoch_entropy_loss
            global_epoch = iteration * self.opt_num_epochs + epoch
            tb_logger.add_scalar('loss/epoch_loss', epoch_loss, global_epoch)
            tb_logger.add_scalar('loss/epoch_value_loss', epoch_value_loss,
                                 global_epoch)
            tb_logger.add_scalar('loss/epoch_surr_loss', epoch_surr_loss,
                                 global_epoch)
            tb_logger.add_scalar('loss/epoch_entropy_loss', epoch_entropy_loss,
                                 global_epoch)

        tb_logger.add_scalar('loss/total_loss',
                             total_loss / self.opt_num_epochs, iteration)
        tb_logger.add_scalar('loss/total_value_loss',
                             total_value_loss / self.opt_num_epochs, iteration)
        tb_logger.add_scalar('loss/total_surr_loss',
                             total_surr_loss / self.opt_num_epochs, iteration)
        tb_logger.add_scalar('loss/total_entropy_loss',
                             total_entropy_loss / self.opt_num_epochs,
                             iteration)

    def ppo_entropy_loss(self, states, actions, advantages, fixed_log_probs,
                         ind):
        log_probs, entropy = self.policy_net.get_log_prob_entropy(
            self.trans_policy(states), actions)
        ratio = torch.exp(log_probs[ind] - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                            1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy[ind].mean()
        return surr_loss, entropy_loss

    def log_optimize_policy(self, iteration, info):
        cfg = self.cfg
        log, log_eval = info['log'], info['log_eval']
        logger, tb_logger = self.logger, self.tb_logger
        log_str = f'{iteration}\tT_sample {info["T_sample"]:.2f}\tT_update {info["T_update"]:.2f}\t' \
                  f'T_eval {info["T_eval"]:.2f}\t' \
                  f'ETA {get_eta_str(iteration, cfg.max_num_iterations, info["T_total"])}\t' \
                  f'train_R_eps {log.avg_episode_reward + self.reward_offset:.2f}\t'\
                  f'eval_R_eps {log_eval.avg_episode_reward + self.reward_offset:.2f}\t{cfg.id}'
        logger.info(log_str)

        self.current_rewards = log_eval.avg_episode_reward + self.reward_offset
        self.current_plans = log_eval.plans
        if log_eval.avg_episode_reward + self.reward_offset > self.best_rewards:
            self.best_rewards = log_eval.avg_episode_reward + self.reward_offset
            self.best_plans = log_eval.plans
            self.save_best_flag = True
        else:
            self.save_best_flag = False

        tb_logger.add_scalar('train/train_R_eps_avg',
                             log.avg_episode_reward + self.reward_offset,
                             iteration)
        tb_logger.add_scalar('train/final_i_rate', log.avg_episode_final_i_rate, iteration)
        tb_logger.add_scalar('train/total_i_rate', log.avg_episode_total_i_rate, iteration)

        tb_logger.add_scalar('eval/eval_R_eps_avg',
                             log_eval.avg_episode_reward + self.reward_offset,
                             iteration)
        tb_logger.add_scalar('eval/final_i_rate', log_eval.avg_episode_final_i_rate, iteration)
        tb_logger.add_scalar('eval/total_i_rate', log_eval.avg_episode_total_i_rate, iteration)


    def  eval_agent(self, num_samples=1, mean_action=True, visualize=True, agent_dict=None):
        t_start = time.time()
        to_test(*self.sample_modules)
        self.env.eval()
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                logger = self.logger_cls(**self.logger_kwargs)
                while logger.num_steps < num_samples:
                    state = self.env.reset(eval=True, agent_dict=agent_dict)
                    info_dict = self.env.get_info()
                    network_id = info_dict['network_id']
                    source = info_dict['source']
                    if visualize:
                        self.env.visualize(save_fig=0,
                                           path=os.path.join(
                                               self.cfg.plan_dir,
                                               f'origin_{network_id}_{source}_{t_start}'))
                    logger.start_episode(self.env)

                    info_plan = dict()
                    episode_success = False
                    for t in tqdm(range(1, 10000), position=0):
                        state_var = tensorfy([state])
                        action = self.policy_net.select_action(
                            state_var, mean_action).numpy()
                        next_state, reward, done, info = self.env.step(
                            action, self.logger)
                        logger.step(self.env, reward, info)

                        # if visualize:
                        #     self.env.visualize(save_fig=True,
                        #                        path=os.path.join(
                        #                            self.cfg.plan_dir,
                        #                            f'step_all_{t:04d}.svg'))
                        if done:
                            episode_success = (reward != self.env.FAILURE_REWARD) and \
                                              (reward != self.env.INTERMEDIATE_REWARD)
                            info_plan = info
                            break
                        state = next_state

                        # self.logger.info(f'reward:{reward}  step:{t:02d}')
                    self.logger.info(f"ffir:{info_plan['ffir']}, ftir:{info_plan['ftir']}")
                    self.logger.info(f"fir:{info_plan['fir']}, tir:{info_plan['tir']}")

                    logger.add_plan(info_plan)
                    logger.end_episode(info_plan)
                    if not episode_success:
                        self.logger.info('Plan fails during eval.')
                    else:
                        if visualize:
                            self.env.visualize(save_fig=0,
                                            path=os.path.join(
                                                self.cfg.plan_dir,
                                                f'final_{network_id}_{source}_{t_start}'))
                logger = self.logger_cls.merge([logger], **self.logger_kwargs)

        self.env.train()
        logger.sample_time = time.time() - t_start
        return logger

    def infer(self,
              num_samples=1,
              mean_action=True,
              visualize=True,
              save_video=False,
              only_road=False):
        t_start = time.time()
        
        r_cut = []
        r_err = []
        for cut_step in range(1):
            r = np.array([])
            agent_dict = None
            # agent_dict = {}
            # agent_dict['cut_ration'] = 0.05 + 0.01 * cut_step
            for t in range(20):
                log_eval = self.eval_agent(num_samples,
                                        mean_action=mean_action,
                                        visualize=visualize, agent_dict=agent_dict)
                logger = self.logger
                cal_eval = (log_eval.avg_episode_full_total_i_rate - log_eval.avg_episode_total_i_rate) / log_eval.avg_episode_full_total_i_rate \
                            if log_eval.avg_episode_full_total_i_rate > 1e-2 else 0
                logger.info(f'eval_{cut_step}_{t}: {cal_eval}')
                if cal_eval > 0e-1:
                    r = np.append(r, cal_eval)
            t_eval = time.time() - t_start
            r_avg = np.mean(r)
            r_std = np.std(r)
            r_cut.append(r_avg)
            r_err.append(r_std)
            logger.info(f'Infer time: {t_eval:.2f}')
            logger.info(f'eval_all: {r_avg, r_std}')
        print(r_cut)
        print(r_err)
        # with open(f'r_cut_{t_start}.txt', 'wb') as f:
        #     f.write(str(r_cut).encode('utf-8'))
        #     f.write('\n')
        #     f.write(str(r_err).encode('utf-8'))





