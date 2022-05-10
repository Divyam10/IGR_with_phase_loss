import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from pyhocon import ConfigFactory
import numpy as np
import argparse
import GPUtil
import torch
import utils.general as utils
from model.sample import Sampler
from model.network import gradient
from scipy.spatial import cKDTree
from utils.plots import plot_surface, plot_cuts
import random
import math
import csv

class ReconstructionRunner:

    def run(self):

        print("running")

        self.data = self.data.cuda()
        # print(self.data.shape)
        # self.data.requires_grad_()

        if self.eval:

            print("evaluating epoch: {0}".format(self.startepoch))
            my_path = os.path.join(self.cur_exp_dir, 'evaluation', str(self.startepoch))

            utils.mkdir_ifnotexists(os.path.join(self.cur_exp_dir, 'evaluation'))
            utils.mkdir_ifnotexists(my_path)
            self.plot_shapes(epoch=self.startepoch, path=my_path, with_cuts=True)
            return

        print("training")

        for epoch in range(self.startepoch +1 , self.nepochs + 1):

            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            
            ## Get Ball Points from a set of all the ball points.
            
            ball_pts = torch.tensor(self.all_ball_points)[indices].reshape((self.points_batch*self.n_ball,3))
            
            # print(ball_pts.shape)
            
            ## Get Normal polints and Sampled X if Normals = True
        
            if self.with_normals:
                norml_pnts = self.data[indices,3:]
                curr_data = self.data[indices, :3]
            mnfld_pnts = ball_pts.cuda().requires_grad_()
            

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                print('saving checkpoint: ', epoch)
                self.save_checkpoints(epoch)
                print('plot validation epoch: ', epoch)
                self.plot_shapes(epoch)

            # change back to train mode
            
            self.network.train()
            self.adjust_learning_rate(epoch)
            
            # Get Omega Points 
            
            nonmnfld_pnts = self.get_omega_smaples(ball_pts.shape[0]).cuda().requires_grad_()
            
            # forward pass

            mnfld_pred = self.network(mnfld_pnts)
            nonmnfld_pred = self.network(nonmnfld_pnts.float())
            
            
            # reconstruction loss

            mnfld_pred = mnfld_pred.reshape((self.points_batch, self.n_ball)) 
            recon_l = (mnfld_pred.mean(axis=1)).abs().mean()
    
            
             # Regularization loss 
            
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # print(nonmnfld_grad.shape)
            d_well = self.double_w(nonmnfld_pred)
            r_loss = ( self.ep  * nonmnfld_grad.norm(2, dim=-1) ** 2 + d_well).mean()
            
            # print(r_loss)
            
            loss = self.lamda*recon_l + r_loss
            
            # normals loss
            if self.with_normals:
            
                u = self.network(curr_data).reshape(-1, 1)
                w = ( self.ep  ** 0.5) * u
                normals_loss = (norml_pnts - w).norm(1, dim=1).mean()
                # print(normals_loss)
                
                loss = loss + self.mu * normals_loss
                
            else:
                
                normals_loss = torch.zeros(1)

            # back propagation

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            if epoch % self.conf.get_int('train.status_frequency') == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tRecon loss: {:.6f}'
                    '\tRecog loss: {:.6f}\tNormals Loss: {:.6f}'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item(), recon_l.item(), r_loss.item(), normals_loss.item()))

    def plot_shapes(self, epoch, path=None, with_cuts=False):
        # plot network validation shapes
        with torch.no_grad():

            self.network.eval()

            if not path:
                path = self.plots_dir

            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))

            pnts = self.data[indices, :3]
            print('----------------------------------')
            plot_surface(with_points=True,
                         points=pnts,
                         decoder=self.network,
                         path=path,
                         epoch=epoch,
                         shapename=self.expname,
                         **self.conf.get_config('plot'))

            if with_cuts:
                plot_cuts(points=pnts,
                          decoder=self.network,
                          path=path,
                          epoch=epoch,
                          near_zero=False)
                
    ## Sampling Omrga Points inside a bounding box 
    def get_omega_smaples(self,bp_size):

        xs = 1.5*((self.x2 - self.x1)*np.random.rand(bp_size) + self.x1)
        ys = 1.5*((self.y2 - self.y1)*np.random.rand(bp_size) + self.y1)
        zs = 1.5*((self.z2 - self.z1)*np.random.rand(bp_size) + self.z1)

        return torch.tensor(np.array(list(zip(xs,ys,zs))))
    
    def double_w(self, s):
        
        return (s ** 2) - 2 * (torch.abs(s)) + 1

    def __init__(self, **kwargs):

        self.home_dir = os.path.abspath(os.pardir)

        # config setting

        if type(kwargs['conf']) == str:
            self.conf_filename = './reconstruction/' + kwargs['conf']
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']

        # GPU settings

        self.GPU_INDEX = kwargs['gpu_index']

        if not self.GPU_INDEX == 'ignore':
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        self.num_of_gpus = torch.cuda.device_count()

        self.eval = kwargs['eval']

        # settings for loading an existing experiment

        if (kwargs['is_continue'] or self.eval) and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(self.home_dir, 'exps', self.expname)):
                timestamps = os.listdir(os.path.join(self.home_dir, 'exps', self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue'] or self.eval

        self.exps_folder_name = 'exps'

        utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name)))

        self.input_file = self.conf.get_string('train.input_path')
        
        self.with_normals = kwargs['with_normals']
        self.data = utils.load_point_cloud_by_file_extension(self.input_file,  self.with_normals)
        
        ### HyperParamters setting 
        
#         self.r = 0.01
#         self.ep = 0.01
#         self.lamda = 0.2
#         self.mu = 0.2
#         self.n_ball = 50
#         self.all_ball_points = []

        self.ep = 0.01
        self.r = kwargs['r']
        self.lamda = kwargs['lamda']
        self.mu = kwargs['mu']
        self.n_ball = kwargs['n_ball']
        
        self.all_ball_points = []
        
        ## Looping over all the points to get corresponding ball points

        for xo,yo,zo in self.data[: , :3]:

            x_sb, y_sb, z_sb = [],[],[]
            temp = []
            
            for i in range(0,self.n_ball):

                theta = random.uniform(*(0, 360))
                phi   = random.uniform(*(0, 360))

                x_sb.append(xo + math.sin(phi)* math.cos(theta)*self.r)
                y_sb.append(yo + math.sin(phi)* math.sin(theta)*self.r)
                z_sb.append(zo + math.cos(phi)*self.r)

            temp = np.column_stack((np.array(x_sb), np.array(y_sb), np.array(z_sb)))
            self.all_ball_points.append(temp)
            
        self.all_ball_points = np.array(self.all_ball_points)
        
        #### Bounding Box
        
        
        self.x1, self.y1, self.z1 = torch.min(self.data[:,:3], axis=0).values.numpy()
        self.x2, self.y2, self.z2 = torch.max(self.data[:,:3], axis=0).values.numpy()

        self.expdir = utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name, self.expname))
        utils.mkdir_ifnotexists(self.expdir)

        if is_continue:
            self.timestamp = timestamp
        else:
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        self.cur_exp_dir = os.path.join(self.expdir, self.timestamp)
        utils.mkdir_ifnotexists(self.cur_exp_dir)

        self.plots_dir = os.path.join(self.cur_exp_dir, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.checkpoints_path = os.path.join(self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        self.nepochs = kwargs['nepochs']

        self.points_batch = kwargs['points_batch']
        
        self.d_in = self.conf.get_int('train.d_in')

        self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in,
                                                                                    **self.conf.get_config(
                                                                                        'network.inputs'))

        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))
        self.weight_decay = self.conf.get_float('train.weight_decay')

        self.startepoch = 0

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": self.weight_decay
                },
            ])

        # if continue load checkpoints

        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.startepoch = saved_model_state['epoch']

    def get_learning_rate_schedules(self, schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self, epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--points_batch', type=int, default=128, help='point batch size')
    parser.add_argument('--nepoch', type=int, default=100000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='setup.conf')
    parser.add_argument('--expname', type=str, default='single_shape')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='continue')
    parser.add_argument('--timestamp', default='latest', type=str)
    parser.add_argument('--checkpoint', default='latest', type=str)
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--lamda',default=0.2, help = 'Reconstruction Loss - Lamda')
    parser.add_argument('--mu',default=0.2, help = 'Regularization Loss - mu')
    parser.add_argument('--r',default=0.01, help = 'Radius to select ball points from')
    parser.add_argument('--n_ball',default=50, help = 'number of ball to be selected')
    parser.add_argument('--with_normals',default=False, action="store_true")
    
    
    args = parser.parse_args()
 
    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = args.gpu

    trainrunner = ReconstructionRunner(
            conf=args.conf,
            points_batch=args.points_batch,
            nepochs=args.nepoch,
            expname=args.expname,
            gpu_index=gpu,
            is_continue=args.is_continue,
            timestamp=args.timestamp,
            checkpoint=args.checkpoint,
            eval=args.eval,
            lamda=args.lamda,
            mu=args.mu,
            r=args.r,
            n_ball=args.n_ball,
            with_normals=args.with_normals
        
    )

    trainrunner.run()
