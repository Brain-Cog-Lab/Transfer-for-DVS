from functools import partial
from torch.nn import functional as F
import torchvision
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.datasets import is_dvs_data


@register_model
class SNN7_tiny(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        assert not is_dvs_data(self.dataset), 'SNN7_tiny only support static datasets now'

        self.feature = nn.Sequential(
            BaseConvModule(3, 16, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(16, 64, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, self.num_classes),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)


@register_model
class SNN5(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if not is_dvs_data(self.dataset):
            init_channel = 3
        else:
            init_channel = 2

        self.feature = nn.Sequential(
            BaseConvModule(init_channel, 16, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
            BaseConvModule(16, 64, kernel_size=(5, 5), padding=(2, 2), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
            BaseConvModule(64, 128, kernel_size=(5, 5), padding=(2, 2), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, self.num_classes),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)


@register_model
class VGG_SNN(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False
        self.TET_loss = kwargs['TET_loss'] if 'TET_loss' in kwargs else False
        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']

        self.feature = nn.Sequential(
            BaseConvModule(2, 64, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, self.num_classes),
        )


    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            if self.TET_loss is True:
                return outputs
            else:
                return sum(outputs) / len(outputs)



@register_model
class Transfer_VGG_SNN(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)
        self.coefficient = nn.ParameterList([nn.Parameter(torch.tensor([1.0]), requires_grad=True) for i in range(self.step)])
        self.dataset = kwargs['dataset']
        self.feature = nn.Sequential(
            BaseConvModule(2, 64, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
        )

        self.rgb_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, self.num_classes),
        )

        self.dvs_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, self.num_classes),
        )

    def forward(self, inputs_rgb, inputs_dvs):
        inputs_rgb = self.encoder(inputs_rgb)
        inputs_dvs = self.encoder(inputs_dvs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs_rgb)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs_rgb_feature, outputs_dvs_feature = [], []
            outputs_rgb, outputs_dvs = [], []
            for t in range(self.step):
                # add encode output to list (firing rate)
                x_rgb = inputs_rgb[t]

                # add feature output to list (membrane potential)
                x_rgb = self.feature(x_rgb)
                outputs_rgb_feature.append(self.feature[-2].node.mem)

                # add fc output to list (firing rate)
                x_rgb = self.rgb_fc(x_rgb)
                outputs_rgb.append(x_rgb)

            self.reset()
            for t in range(self.step):
                x_dvs = inputs_dvs[t]
                x_dvs = self.feature(x_dvs)
                outputs_dvs_feature.append(self.feature[-2].node.mem)
                x_dvs = self.dvs_fc(x_dvs)
                outputs_dvs.append(x_dvs)

            return outputs_rgb_feature, outputs_dvs_feature, outputs_rgb, outputs_dvs
            # return sum(inputs_rgb_encode) / len(inputs_rgb_encode), sum(inputs_dvs_encode) / len(inputs_dvs_encode), \
            #        sum(outputs_rgb_feature) / len(outputs_rgb_feature), sum(outputs_dvs_feature) / len(outputs_dvs_feature), \
            #        outputs_rgb, outputs_dvs


@register_model
class NMNIST_SNN(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes
        self.TET_loss = kwargs['TET_loss'] if 'TET_loss' in kwargs else False
        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']

        self.feature = nn.Sequential(
            BaseConvModule(2, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128 * 4 * 4),
            partial(self.node, **kwargs)(),
            nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 100),
            partial(self.node, **kwargs)(),
            VotingLayer(10)
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            if self.TET_loss is True:
                return outputs
            else:
                return sum(outputs) / len(outputs)


@register_model
class Transfer_NMNIST_SNN(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes
        self.coefficient = nn.ParameterList(
            [nn.Parameter(torch.tensor([1.0]), requires_grad=True) for i in range(self.step)])
        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        self.feature = nn.Sequential(
            BaseConvModule(2, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128 * 4 * 4),
            partial(self.node, **kwargs)(),
            nn.Dropout(0.5),
        )

        self.rgb_fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 100),
            partial(self.node, **kwargs)(),
            VotingLayer(10)
        )

        self.dvs_fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 100),
            partial(self.node, **kwargs)(),
            VotingLayer(10)
        )

    def forward(self, inputs_rgb, inputs_dvs):
        inputs_rgb = self.encoder(inputs_rgb)
        inputs_dvs = self.encoder(inputs_dvs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs_rgb)
            outputs_rgb, outputs_dvs = [], []
            outputs_rgb_feature = self.feature[-2].mem_collect
            x = self.rgb_fc(x)
            outputs_rgb = rearrange(x, '(t b) c -> t b c', t=self.step)
            self.reset()

            x = self.feature(inputs_dvs)
            outputs_dvs_feature = self.feature[-2].mem_collect
            x = self.dvs_fc(x)
            outputs_dvs = rearrange(x, '(t b) c -> t b c', t=self.step)
            return outputs_rgb_feature, outputs_dvs_feature, outputs_rgb, outputs_dvs

        else:
            outputs_rgb_feature, outputs_dvs_feature = [], []
            outputs_rgb, outputs_dvs = [], []
            for t in range(self.step):
                # add encode output to list (firing rate)
                x_rgb = inputs_rgb[t]

                # add feature output to list (membrane potential)
                x_rgb = self.feature(x_rgb)
                outputs_rgb_feature.append(self.feature[-2].mem)

                # add fc output to list (firing rate)
                x_rgb = self.rgb_fc(x_rgb)
                outputs_rgb.append(x_rgb)

            self.reset()
            for t in range(self.step):
                x_dvs = inputs_dvs[t]
                x_dvs = self.feature(x_dvs)
                outputs_dvs_feature.append(self.feature[-2].mem)
                x_dvs = self.dvs_fc(x_dvs)
                outputs_dvs.append(x_dvs)

            return outputs_rgb_feature, outputs_dvs_feature, outputs_rgb, outputs_dvs