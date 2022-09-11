import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (ModelCheckpoint, ModelPruning,
                                         QuantizationAwareTraining)
from torchinfo import summary
from torchvision.utils import save_image

import utils
from dataset import DataModule
from diffaug import DiffAugment
from model import Discriminator, Generator


class GAN(pl.LightningModule):
    def __init__(self, generator, discriminator, **kwargs):
        super().__init__()
        # https://github.com/Lightning-AI/lightning/issues/2909
        self.save_hyperparameters()
        # base
        self.name = kwargs['name']
        self.use_aim = kwargs['use_aim']
        self.generator = generator
        self.discriminator = discriminator
        # aug
        self.aug_level = kwargs['aug_level']
        self.aug_type = kwargs['aug_type']
        self.aug_prob = kwargs['aug_prob']
        # train
        self.loss_type = kwargs['loss_type']
        self.smoothing_value = kwargs['smoothing_value']
        self.z_distibution = kwargs['z_distibution']
        self.save_freq = kwargs['save_freq']
        self.g_lr = kwargs['g_lr']
        self.d_lr = kwargs['d_lr']
        # wgan specific
        self.n_critics = kwargs['n_critics']
        self.lambda_gp = kwargs['lambda_gp']

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_size = batch.size(0)

        # discriminator
        if optimizer_idx == 0:
            if self.aug_level == 2 or (self.aug_level == 1 and torch.rand(1) < self.aug_prob):
                batch = DiffAugment(batch, self.aug_type)

            d_real = self.discriminator(batch)
            input_z = self._create_noise(batch_size)
            g_real = self.generator(input_z)

            if self.aug_level >= 2:
                g_real = DiffAugment(g_real, self.aug_type)
            d_fake = self.discriminator(g_real)

            if self.loss_type == 0:
                real_label = torch.full((batch_size, ), 1.0 - self.smoothing_value,
                                        dtype=batch.dtype, device=self.device, requires_grad=False)
                fake_label = torch.full((batch_size, ), 0.0 + self.smoothing_value,
                                        dtype=batch.dtype, device=self.device, requires_grad=False)

                d_loss_real = F.binary_cross_entropy(d_real, real_label)
                d_loss_fake = F.binary_cross_entropy(d_fake, fake_label)
                d_loss = (d_loss_real + d_loss_fake) / 2

                self.log('d_loss', d_loss, logger=True)
                self.log('d_x_loss', d_loss_real, logger=True)
                self.log('d_g_z_loss', d_loss_fake, logger=True)
                return {'loss': d_loss, 'loss_type': 'd_loss', 'D(x)': d_loss_real, 'D(G(z))': d_loss_fake}
            else:  # 1
                d_loss = d_fake.mean() - d_real.mean() + \
                    self._gradient_penalty(batch.data, g_real.data)

                self.log('d_loss', d_loss, logger=True)
                return {'loss': d_loss, 'loss_type': 'd_loss'}

        # generator
        if optimizer_idx == 1:
            input_z = self._create_noise(batch_size)
            g_real = self(input_z)
            if self.aug_level >= 2:
                g_real = DiffAugment(g_real, self.aug_type)
            d_fake = self.discriminator(g_real)

            # reset rand_result
            self.rand_result = torch.rand(1)

            if self.loss_type == 0:
                fake_label = torch.full((batch_size, ), 1.0 - self.smoothing_value,
                                        dtype=batch.dtype, device=self.device, requires_grad=False)

                g_loss = F.binary_cross_entropy(d_fake, fake_label)
            else:  # 1
                g_loss = -d_fake.mean()

            self.log('g_loss', g_loss, logger=True)
            return {'loss': g_loss, 'loss_type': 'g_loss'}

    def _create_noise(self, batch_size) -> torch.Tensor:
        if self.z_distibution == 0:
            return torch.rand(batch_size, self.generator.n_z, 1, 1, device=self.device) * 2 - 1
        else:  # 1
            return torch.rand(batch_size, self.generator.n_z, 1, 1, device=self.device)

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(
            real_data.shape[0], 1, 1, 1, requires_grad=True, device=self.device
        )
        interpolated = alpha * real_data + (1 - alpha) * generated_data

        # Calculate probability of interpolated examples
        proba_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=proba_interpolated, inputs=interpolated,
            grad_outputs=torch.ones(proba_interpolated.size(), device=self.device),
            create_graph=True, retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        return self.lambda_gp * ((gradients_norm - 1)**2).mean()

    def on_train_epoch_end(self):
        if self.current_epoch % self.save_freq == 0:
            # Only generate image
            input_z = self._create_noise(64)  # TO DO: get batch size
            images = self.predict_step(input_z, 0)
            save_image(images, f'./{self.name}/result/{self.current_epoch:05d}.png')

            if self.use_aim:
                from aim import Image
                img = Image(
                    f'./{self.name}/result/{self.current_epoch:05d}.png')
                self.logger.experiment.track(
                    img, name='images', epoch=self.current_epoch)

    def training_epoch_end(self, outputs):
        d_loss = []
        g_loss = []
        gx_loss = []
        gdz_loss = []

        for output in outputs:
            if output[0]['loss_type'] == 'd_loss':
                d_loss.append(output[0]['loss'])
                if self.loss_type == 0:
                    gx_loss.append(output[0]['D(x)'])
                    gdz_loss.append(output[0]['D(G(z))'])
            else:
                g_loss.append(output[0]['loss'])

        d_loss = torch.stack(d_loss).mean()
        g_loss = torch.stack(g_loss).mean()

        if self.loss_type == 0:
            gx_loss = torch.stack(gx_loss).mean()
            gdz_loss = torch.stack(gdz_loss).mean()

        if self.loss_type == 0:
            print(f'Epoch #{self.current_epoch:05d} | d_loss: {d_loss:04f} | g_loss: {g_loss:04f} | D(x): {gx_loss:04f} | D(G(z)): {gdz_loss:04f}')
        else:
            print(f'Epoch #{self.current_epoch:05d} | d_loss: {d_loss:04f} | g_loss: {g_loss:04f}')

    def predict_step(self, batch, batch_idx):
        images = self(batch)
        images = (images+1)/2  # Reminder: tanh output range is -1 to 1

        return images

    def configure_optimizers(self):
        if self.loss_type == 0:
            d_optimizer = torch.optim.Adam(self.discriminator.parameters(
            ), lr=self.d_lr, betas=(0.5, 0.999), weight_decay=0.001)
            g_optimizer = torch.optim.Adam(
                self.generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))

            return [
                {'optimizer': d_optimizer, 'frequency': 1},
                {'optimizer': g_optimizer, 'frequency': 1},
            ]
        else:  # 1
            d_optimizer = torch.optim.Adam(self.discriminator.parameters(
            ), lr=self.d_lr, betas=(0.0, 0.999), weight_decay=0.001)
            g_optimizer = torch.optim.Adam(
                self.generator.parameters(), lr=self.g_lr, betas=(0.0, 0.999))

            return [
                {'optimizer': d_optimizer, 'frequency': self.n_critics},
                {'optimizer': g_optimizer, 'frequency': 1},
            ]


def show_model_summary(net_g, net_d, args):
    column = ["input_size", "output_size", "num_params"]
    width, height = args.im_size
    if args.transparent:
        channel = 4
    else:
        channel = 3
    summary(
        net_g, input_size=(2, args.n_z, 1, 1),
        verbose=1, col_names=column, device='cpu'
    )
    summary(
        net_d, input_size=(2, channel, height, width),
        verbose=1, col_names=column, device='cpu'
    )


def setup(args):
    net_g = Generator(args)
    net_d = Discriminator(args)
    net = GAN(net_g, net_d, **vars(args))

    return net_g, net_d, net


def train(args):
    # Environment setup
    utils.check_path_exist(f'./{args.name}/result')
    if len(args.im_size) == 1:
        args.im_size = [args.im_size[0], args.im_size[0]]
    pl.seed_everything(42, workers=True)

    callback_list: list = []
    logger = None

    # Create model and datamodule
    net_g, net_d, net = setup(args)
    dm = DataModule(args)
    show_model_summary(net_g, net_d, args)

    # Configure callback
    if args.prune:
        callback_list.append(
            ModelPruning(
                'random_unstructured',
                amount=0.3,
            )
        )
    if args.quantize:
        callback_list.append(
            QuantizationAwareTraining(
                'qnnpack', 'histogram', input_compatible=True)
        )
    if args.use_aim:
        from aim.pytorch_lightning import AimLogger
        logger = AimLogger(experiment=args.name)

    callback_list.append(
        ModelCheckpoint(
            dirpath=f'./{args.name}/checkpoints',
            filename='GAN_{epoch}_{g_loss:.4f}_{d_loss:.4f}',
            every_n_epochs=args.save_freq,
            save_top_k=-1,
            save_last=False,
        )
    )

    # Setup trainer, than train GAN
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices='auto',
        max_epochs=args.n_epochs,
        logger=logger,
        default_root_dir=f'./{args.name}/checkpoints',
        enable_checkpointing=True,  # save last
        callbacks=callback_list,
        precision=32,  # CUDA only
        profiler='simple',  # check bottleneck
        enable_model_summary=False,
        enable_progress_bar=False
    )
    ckpt_path = utils.get_ckpt(args.name, args.ckpt_name)
    trainer.fit(model=net, datamodule=dm, ckpt_path=ckpt_path)


def generate(args):
    # TODO: reduce GPU memory usage during generation
    utils.check_path_exist(args.output_path)
    ckpt_path = utils.get_ckpt(args.name, args.ckpt_name)
    torch_dict = torch.load(ckpt_path)

    net_g = torch_dict['hyper_parameters']['generator'].cuda()
    n_z = torch_dict['hyper_parameters']['n_z']
    batch_size = torch_dict['hyper_parameters']['batch_size']

    input_z = torch.rand(args.total_image, n_z, 1, 1).cuda() * 2 - 1
    image_batches = []
    from tqdm import tqdm
    for i in tqdm(range(0, args.total_image, batch_size)):
        image_batch = net_g(input_z[i:i+batch_size])
        image_batches.extend(image_batch.cpu())

    image_batches = [(img+1)/2 for img in image_batches]

    for idx, image_batch in enumerate(image_batches):
        save_image(image_batch, f'{args.output_path}/{idx}.png')


def to_onnx(args):
    ckpt_path = utils.get_ckpt(args.name, args.ckpt_name)

    torch_dict = torch.load(ckpt_path)

    net_g = torch_dict['hyper_parameters']['generator'].cpu()
    n_z = torch_dict['hyper_parameters']['n_z']
    input_z = torch.rand(1, n_z, 1, 1) * 2 - 1
    name = torch_dict['hyper_parameters']['name']

    # assume ckpt extension always .ckpt
    onnx_filename = name + "_" + os.path.basename(ckpt_path)[:-5] + '.onnx'

    # can't export with opset_version 13
    torch.onnx.export(
        net_g, input_z, onnx_filename, verbose=True, opset_version=12,
        input_names=['random_input'], output_names=['image_output']
    )
