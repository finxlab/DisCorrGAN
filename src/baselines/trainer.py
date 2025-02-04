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

class GANTrainer:
    def __init__(self, D, G, train_dl, config, **kwargs):                
        self.G = G
        self.D = D        
        self.config = config                                                        
        self.train_dl = train_dl        
        self.device = self.config.device

        # Directory settings
        self.full_name = (
            f'{config.n_epochs}_{config.batch_size}_Glr_{config.lr_G}_Dlr_{config.lr_D}_hidden_dim_{config.hidden_dim}'
            f'_corr_loss_{config.corr_loss_type}_corr_weight_{config.corr_weight}_n_critic_{config.D_steps_per_G_step}_gp_{config.lambda_gp}_noise_{config.noise_dim}_Adam_drop_0.1_PReLU_16'
        )
        self.results_dir = f'./results/models/{self.full_name}/'        
        os.makedirs(self.results_dir, exist_ok=True)                    

        # Initialize optimizers and learning rate schedulers
        self.G_optimizer = [torch.optim.Adam(G.generators[i].parameters(), lr=config.lr_G, betas=(0.5, 0.999)) for i in range(config.n_vars)]
        self.D_optimizer = [torch.optim.Adam(D.discriminators[i].parameters(), lr=config.lr_D, betas=(0.5, 0.999)) for i in range(config.n_vars)]
        self.G_scheduler = [lr_scheduler.StepLR(self.G_optimizer[i], step_size=10, gamma=0.95) for i in range(config.n_vars)]
        self.D_scheduler = [lr_scheduler.StepLR(self.D_optimizer[i], step_size=10, gamma=0.95) for i in range(config.n_vars)]            
    
    def fit(self):        
        self.G.to(self.device)
        self.D.to(self.device)
        wandb.init(project="MVFIT_GAN", config=config)

        for epoch in tqdm(range(self.config.n_epochs)):            
            for n, (real) in enumerate(self.train_dl):
                real_batch = real[0].transpose(1, 2).to(self.device)
                self.step(real_batch, n)        
                        
            for i in range(config.n_vars):
                    self.G_scheduler[i].step()
                    self.D_scheduler[i].step()                     
                    
            # Save model every few epochs
            if epoch % 5 == 0 and epoch >= 50:                
                torch.save(self.G.state_dict(), f'{self.results_dir}/Generator_{epoch}.pt')                
    
    def step(self, real, n):
        batch_size = real.shape[0]        
        with torch.no_grad():
            fake = self.G(batch_size, self.config.n_steps, self.device)                

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

        # Train Generator
        if n % self.config.D_steps_per_G_step == 0:                                                
            for i in range(self.config.n_vars):                   
                self.G_optimizer[i].zero_grad()                
                gen_loss = 0 
                fake = self.G(batch_size, self.config.n_steps, self.device)                                           
                fake_pred = self.D(fake[:, i:i+1, :], i)

                gen_loss = -torch.mean(fake_pred)                   
                corr_loss, _ = correlation_loss(fake=fake, real=real, corr_loss_type=self.config.corr_loss_type)                                                               
                
                gen_loss = gen_loss + self.config.corr_weight * corr_loss
                gen_loss.backward(retain_graph=True)                            
                self.G_optimizer[i].step()
                
                # Log generator and cross-correlation loss for this variable
                wandb.log({f"Generator Loss (Var {i})": gen_loss.item()})
                if i == 0:
                    wandb.log({f"Cross-Correlation Loss (Var {i})": corr_loss.item()})                
                
                
                
                                    
