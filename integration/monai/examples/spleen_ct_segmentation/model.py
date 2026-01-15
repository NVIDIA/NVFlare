from monai.networks.nets import UNet

class FLUNet(UNet):
    def __init__(self, spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=[
                16,
                32,
                64,
                128,
                256
            ],
            strides=[
                2,
                2,
                2,
                2
            ],
        num_res_units=2,
        norm="batch"):

        # Store configuration for JobAPI
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.num_res_units = num_res_units
        self.norm = norm

        super().__init__(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, channels=channels, strides=strides, num_res_units=num_res_units, norm=norm)
