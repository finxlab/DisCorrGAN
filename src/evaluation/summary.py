from src.evaluation.loss import *
from src.evaluation.strategies import *
import pandas as pd
from src.utils import *
from tqdm import tqdm


STRATEGIES ={
             'equal_weight': EqualWeightPortfolioStrategy,             
             'mean_reversion': MeanReversionStrategy,
             'trend_following': TrendFollowingStrategy,
             'vol_trading': VolatilityTradingStrategy
             }

def full_evaluation(fake_dataset, real_dataset, config, **kwargs):
    ec = EvaluationComponent(config, fake_dataset, real_dataset, **kwargs)
    summary_dict = ec.eval_summary()
    return summary_dict


class EvaluationComponent(object):
    '''
    Evaluation component for evaluation metrics according to config
    '''

    def __init__(self, config, fake_dataset, real_data, **kwargs):
        self.config = config
        self.fake_data = fake_dataset
        self.kwargs = kwargs
        self.n_eval = self.config.Evaluation.n_eval

        if 'seed' in kwargs:
            self.seed = kwargs['seed']
        elif 'seed' in config:
            self.seed = config.seed
        else:
            self.seed = 52

        self.real_data = real_data
        self.dim = self.real_data.shape[-1]

        self.sample_size = min(self.real_data.shape[0], self.config.Evaluation.batch_size)

        set_seed(self.config.seed)
        self.data_set = self.get_data(n=self.n_eval)

        strat_name = kwargs.get('strat_name', 'equal_weight')
        self.strat = STRATEGIES[strat_name]()
        self.metrics_group = {            
            'tail_scores': ['var', 'es'],
            'trading_strat_scores': ['max_drawback', 'cumulative_pnl']
        }

    def get_data(self, n=1):
        batch_size = int(self.sample_size)

        idx_all = torch.randint(self.real_data.shape[0], (batch_size * n,))
        idx_all_test = torch.randint(self.fake_data.shape[0], (batch_size * n,))
        data = {}
        for i in range(n):
            idx = idx_all[i * batch_size:(i + 1) * batch_size]
            # idx = torch.randint(real_data.shape[0], (sample_size,))
            real = self.real_data[idx]
            # idx = idx_all_test[i * batch_size:(i + 1) * batch_size]
            fake = self.fake_data[idx]
            data.update({i:
                {
                    'real': real,
                    'fake': fake
                }
            })
        return data

    def eval_summary(self):
        metrics = self.config.Evaluation.metrics_enabled
        
        scores = {metric: {'abs': [], 'rel': []} for metric in metrics}
        summary = {}

        for grp in self.metrics_group.keys():
            metrics_in_group = [m for m in metrics if m in self.metrics_group[grp]]
            if len(metrics_in_group) == 0:
                continue

            for metric in metrics_in_group:
                eval_func = getattr(self, metric)
                
                # 임시 리스트 (각 metric별로 abs/rel 스코어를 저장)
                tmp_abs = []
                tmp_rel = []

                for i in range(self.n_eval):
                    real = self.data_set[i]['real']
                    fake = self.data_set[i]['fake']

                    # 예: mean reversion 같은 전략의 PnL 등 구하기
                    pnl_real = self.strat.get_pnl_trajectory(real)
                    pnl_fake = self.strat.get_pnl_trajectory(fake)

                    # (abs_loss_tensor, rel_loss_tensor) 형태로 반환된다고 가정
                    abs_loss_tensor, rel_loss_tensor = eval_func(pnl_real, pnl_fake)
                    # 텐서 -> 넘파이로 변환
                    abs_loss_numpy = to_numpy(abs_loss_tensor)
                    rel_loss_numpy = to_numpy(rel_loss_tensor)

                    # 각 이터레이션마다 "평균값"만 사용하거나, 전체 값을 붙일 수도 있음
                    tmp_abs.append(abs_loss_numpy.mean())
                    tmp_rel.append(rel_loss_numpy.mean())

                # 이제 tmp_abs, tmp_rel에 n_eval개 만큼의 평균값이 들어있으니,
                # 이것들에 대해 최종 mean/std 계산
                abs_mean = np.mean(tmp_abs)
                abs_std = np.std(tmp_abs)
                rel_mean = np.mean(tmp_rel)
                rel_std = np.std(tmp_rel)

                # summary에 기록
                summary[f'{metric}_abs_mean'] = abs_mean
                summary[f'{metric}_abs_std'] = abs_std
                summary[f'{metric}_rel_mean'] = rel_mean
                summary[f'{metric}_rel_std'] = rel_std

        # DataFrame으로 만들어 출력하거나 return
        df = pd.DataFrame([summary])
        return summary

    def max_drawback(self, real, fake):
        loss = to_numpy(MaxDrawbackLoss(real, name='max_drawback_loss')(fake))
        return loss

    def cumulative_pnl(self, real, fake):
        loss = to_numpy(CumulativePnLLoss(real, name='cum_pnl_loss')(fake))
        return loss
    
    def var(self, real, fake):
        ecfg = self.config.Evaluation.TestMetrics.var    
        losses = VARLoss(real.unsqueeze(2), name='var_loss', alpha=ecfg.alpha)(fake.unsqueeze(2))        
        abs_loss = losses['abs_loss']
        rel_loss = losses['rel_loss']  
        return (abs_loss, rel_loss)
    def es(self, real, fake):
        ecfg = self.config.Evaluation.TestMetrics.es   
        losses = ESLoss(real.unsqueeze(2), name='es_loss', alpha=ecfg.alpha)(fake.unsqueeze(2))        
        abs_loss = losses['abs_loss']
        rel_loss = losses['rel_loss']  
        return (abs_loss, rel_loss)
