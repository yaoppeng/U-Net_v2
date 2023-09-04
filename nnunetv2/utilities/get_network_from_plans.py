from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
# from nnunetv2.training.network.dynamic_network_architectures.\
#     dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn


def get_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True,
                           base_ch=16,
                           block="FusedMBConv",
                           use_my_unet=True,
                           setting=0):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accomodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    # network class name!!

    # a = configuration_manager.configuration
    if configuration_manager.configuration["nnUNet_UNet"]:
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
    else:
        if configuration_manager.configuration["my_net_class"] == "UNet":
            print(f"setting: {setting}".center(40, "="))
            if setting == 0:
                from nnunetv2.training.network.model.dim2.unet import UNet
            elif setting == 1:
                from nnunetv2.training.network.model.dim2. \
                    unet_reverse_attention import UNet
            elif setting == 2:
                from nnunetv2.training.network.model.dim2. \
                    unet_reverse_attention_1 import UNet
            else:
                raise NotImplementedError("not implemented.")
            model = UNet(in_ch=3, num_classes=2, block=block,
                         base_ch=base_ch, use_my_net=use_my_unet)

        elif configuration_manager.configuration["my_net_class"] == "Res2UNet":
            from nnunetv2.training.network.model.dim2.res2net.res2unetv2 import Res2Network
            model = Res2Network(channel=32, n_classes=2)   # .cuda()
        elif configuration_manager.configuration["my_net_class"] == "ternaus_unet":
            from nnunetv2.training.network.model.dim2.ternaus.ternaus_unet import UNet16
            model = UNet16(num_classes=2, num_filters=32, pretrained=True,
                           deep_supervision=True)  # .cuda()
        elif configuration_manager.configuration["my_net_class"] == "pvt_1":
            from nnunetv2.training.network.model.dim2.pvt.pvt_1 import PVTNetwork_1
            model = PVTNetwork_1(n_classes=2, deep_supervision=True)
        elif configuration_manager.configuration["my_net_class"] == "pvt_2":
            from nnunetv2.training.network.model.dim2.pvt.pvt_2 import PVTNetwork_2
            model = PVTNetwork_2(n_classes=2, deep_supervision=True)
        elif configuration_manager.configuration["my_net_class"] == "pvt_3":
            from nnunetv2.training.network.model.dim2.pvt.pvt_3 import PVTNetwork_3
            model = PVTNetwork_3(n_classes=2, deep_supervision=True)
        elif configuration_manager.configuration["my_net_class"] == "pvt_unet":
            from nnunetv2.training.network.model.dim2.pvt.pvt_unet import PVTUNet
            model = PVTUNet(n_classes=2, deep_supervision=False)
        elif configuration_manager.configuration["my_net_class"] == "pvt_unetplus":
            from nnunetv2.training.network.model.dim2.pvt.pvt_unetplusplus import PVTUNetPlus
            model = PVTUNetPlus(n_classes=2, deep_supervision=False)
        elif configuration_manager.configuration["my_net_class"] == "pvt_4":
            from nnunetv2.training.network.model.dim2.pvt.pvt_4 import PVTNetwork_4
            model = PVTNetwork_4(n_classes=2, deep_supervision=False)
        else:
            raise NotImplementedError("not implemented")

        print(f"model: {model}")

    print(f"{model.__class__}".center(100, "="))

    # model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model
