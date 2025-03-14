import torch
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm
import os, wandb
from os import path as pt
from src.utils import *
from src.evaluation.strategies import *

config_dir = pt.join("configs/config.yaml")
config = (load_config(config_dir))

# def linear_rampup(current_epoch, rampup_epochs, final_value):
#     """
#     current_epoch: 현재 epoch 번호
#     rampup_epochs: ramp-up을 적용할 총 epoch 수
#     final_value: ramp-up이 완료된 후 최종적으로 사용할 가중치 값

#     return: 현재 epoch에 적용할 가중치 값 (0 ~ final_value)
#     """
#     init_epoch = 150
#     if current_epoch < init_epoch:
#         return 0.0
#     elif current_epoch < rampup_epochs:
#         return final_value * (current_epoch - init_epoch) / (rampup_epochs - init_epoch)
#         #return final_value * (current_epoch / rampup_epochs - 50)
#     else:
#         return final_value
    
    
class GANTrainer:
    def __init__(self, D, G, train_dl, config, **kwargs):                
        self.G = G
        self.D = D        
        self.config = config                                                        
        self.train_dl = train_dl        
        self.device = self.config.device

        # Directory settings
        self.full_name = (
            f'{config.n_epochs}_{config.batch_size}_Glr_{config.lr_G}_Dlr_{config.lr_D}_hidden_dim_{config.hidden_dim}_n_steps_{config.n_steps}'
            f'_corr_loss_{config.corr_loss_type}_corr_weight_{config.corr_weight}_f_epoch_{config.rampup_epochs}_n_critic_{config.D_steps_per_G_step}_gp_{config.lambda_gp}_noise_{config.noise_dim}_Adam_drop_{config.G_dropout}_{config.D_dropout}_Scale0.2_rampupX_Adam0_LN'
        )
        self.results_dir = f'./results/models/{self.full_name}/'        
        os.makedirs(self.results_dir, exist_ok=True)                    

        # Initialize optimizers and learning rate schedulers
        self.G_optimizer = [torch.optim.Adam(G.generators[i].parameters(), lr=config.lr_G, betas=(0.0, 0.9)) for i in range(config.n_vars)]
        self.D_optimizer = [torch.optim.Adam(D.discriminators[i].parameters(), lr=config.lr_D, betas=(0.0, 0.9)) for i in range(config.n_vars)]
        T_max_val = config.n_epochs  # 예: 전체 epochs
        eta_min_val = 5e-6          # 예: 1e-6 정도로 설정
        self.G_scheduler = [lr_scheduler.CosineAnnealingLR(self.G_optimizer[i], T_max=T_max_val, eta_min=eta_min_val) for i in range(config.n_vars)]
        self.D_scheduler = [lr_scheduler.CosineAnnealingLR(self.D_optimizer[i], T_max=T_max_val, eta_min=eta_min_val) for i in range(config.n_vars)]
    
    def fit(self):        
        self.G.to(self.device)
        self.D.to(self.device)
        wandb.init(project="MVFIT_GAN", config=config)

        for epoch in tqdm(range(self.config.n_epochs)):
            # 매 epoch마다 현재의 correlation 가중치를 계산 (ramp-up 적용)          
            #corr_weight = linear_rampup(epoch, config.rampup_epochs, config.corr_weight)            
            corr_weight = config.corr_weight
                        
            for n, (real) in enumerate(self.train_dl):
                real_batch = real[0].transpose(1, 2).to(self.device)
                self.step(real_batch, n, corr_weight)        

            for i in range(config.n_vars):
                self.G_scheduler[i].step()
                self.D_scheduler[i].step()                     
                    
            # Save model every few epochs
            if epoch % 2 == 0 and epoch >= 50:                
                for i in range(self.config.n_vars):
                    torch.save(self.G.generators[i].state_dict(), f'{self.results_dir}/Generator_{epoch}_var_{i}.pt')
    
    def step(self, real, n, corr_weight):
        batch_size = real.shape[0]       
        noise = torch.randn(batch_size, self.config.noise_dim, self.config.n_steps, device=self.device)
 
        with torch.no_grad():
            fake = self.G(noise)                

        # Train Discriminator for each feature
        for i in range(self.config.n_vars):
            self.D_optimizer[i].zero_grad()
            real_pred = self.D(real[:, i:i+1, :], i)
            fake_pred = self.D(fake[:, i:i+1, :], i)
            loss_D = -torch.mean(real_pred) + torch.mean(fake_pred)

            if self.config.gp:
                gradient_penalty = compute_gradient_penalty(self.D, real[:, i:i+1, :], fake[:, i:i+1, :], i)                    
                loss_D = loss_D + config.lambda_gp * gradient_penalty                                

            #loss_D.backward(retain_graph=True)
            loss_D.backward()
            self.D_optimizer[i].step()
            wandb.log({f"Discriminator Loss (Var {i})": loss_D.item()})
            
        # if n % self.config.D_steps_per_G_step == 0:        
        #     for g_opt in self.G_optimizer:
        #         g_opt.zero_grad()        
        #     fake = self.G(noise)
            
        #     total_adv_loss = 0.
        #     for i in range(self.config.n_vars):
        #         fake_pred_i = self.D(fake[:, i:i+1, :], i)
        #         total_adv_loss += -torch.mean(fake_pred_i)

            
        #     corr_loss_val, _ = correlation_loss(fake, real, self.config.corr_loss_type)
        #     dynamic_corr_weight = self.config.corr_weight * corr_loss_val.detach()            
        #     total_gen_loss = total_adv_loss + dynamic_corr_weight * corr_loss_val

        #     # (B5) backward once
        #     total_gen_loss.backward()

        #     # (B6) step all G optimizers
        #     for i, g_opt in enumerate(self.G_optimizer):
        #         g_opt.step()
                
        #     wandb.log({
        #         "Adv G Loss (sum)": total_adv_loss.item(),
        #         "Corr Loss": corr_loss_val.item()
        #     })
        
        
        # (C) Generator step: correlation loss만 반영하는 별도 단계
        if n % self.config.D_steps_per_G_step == 0:
            # G를 다시 forward해서 correlation loss 계산
            for g_opt in self.G_optimizer:
                g_opt.zero_grad()
            fake = self.G(noise)
            corr_loss_val, _ = correlation_loss(fake, real, self.config.corr_loss_type)                        
            
            #dynamic_corr_weight = self.config.corr_weight * corr_loss_val.detach()
            
            loss_corr_final = corr_weight * corr_loss_val
            loss_corr_final.backward()
            
            for g_opt in self.G_optimizer:
                g_opt.step()
                
            wandb.log({                
                "Corr Loss": corr_loss_val.item()
            })
            
        # (B) Generator step: adversarial loss만 반영
        if n % self.config.D_steps_per_G_step == 0:
            for g_opt in self.G_optimizer:
                g_opt.zero_grad()

            fake = self.G(noise)
            total_adv_loss = 0
            for i in range(self.config.n_vars):
                fake_pred_i = self.D(fake[:, i:i+1, :], i)
                total_adv_loss += -torch.mean(fake_pred_i)

            total_adv_loss.backward()
            for g_opt in self.G_optimizer:
                g_opt.step()
                
            # Logging
            wandb.log({
                "Adv G Loss (sum)": total_adv_loss.item(),
                #"Corr Loss": corr_loss_val.item()
            })
            
        
        