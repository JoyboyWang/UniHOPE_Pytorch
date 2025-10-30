class HONet(nn.Module):
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1,
                 transformer_depth=1, transformer_head=8,
                 mano_layer=None, mano_neurons=[1024, 512], coord_change_mat=None,
                 reg_object=True, pretrained=True):

        super(HONet, self).__init__()

        self.out_res = roi_res

        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
                                      stacks=stacks, channels=channels, blocks=blocks)
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        # mano branch
        self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.num_feat_out,
                                        mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)
        # object head
        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels//2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        # CR blocks
        self.transformer_obj = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth, num_heads=transformer_head)
        self.transformer_hand = Transformer(inp_res=roi_res, dim=channels*2,depth=transformer_depth, num_heads=transformer_head)

        self.hand_head.apply(init_weights)
        self.hand_encoder.apply(init_weights)
        self.mano_branch.apply(init_weights)
        self.transformer_obj.apply(init_weights)
        self.transformer_hand.apply(init_weights)
        self.obj_head.apply(init_weights)



    def net_forward(self, imgs, bbox_hand, bbox_obj,mano_params=None, roots3d=None):
        batch = self.new_method(imgs)

        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
        msk_inter = ((inter_bottomRight-inter_topLeft > 0).sum(dim=1)) == 2
        # P2 from FPN Network
        P2_h,P2_o = self.base_net(imgs)
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        # 4 here is the downscale size in FPN network(P2)
        x_hand = ops.roi_align(P2_h, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand

        x_obj = ops.roi_align(P2_o, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand

        # obj forward
        if self.reg_object:
            roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
            roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)

            y = ops.roi_align(P2_o, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # obj



            z_x = ops.roi_align(P2_h, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # intersection

    
            z_x = msk_inter[:, None, None, None] * z_x

            #print(3)

            hand_obj = torch.cat([x_hand,x_obj.detach()],dim=1)
            hand_obj = self.transformer_hand(hand_obj,hand_obj)

            y = self.transformer_obj(y, z_x.detach())

            out_fm = self.obj_head(y)
            preds_obj = self.obj_reorgLayer(out_fm)
        else:
            preds_obj = None

        hand = hand_obj[:,0:256,:,:]
        #hand forward

        out_hm, encoding, preds_joints = self.hand_head(hand)

        mano_encoding = self.hand_encoder(out_hm, encoding)

        pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, mano_params=mano_params, roots3d=roots3d)

        return preds_joints, pred_mano_results, gt_mano_results, preds_obj

    def new_method(self, imgs):
        batch = imgs.shape[0]
        return batch

    def forward(self, imgs, bbox_hand, bbox_obj,mano_params=None, roots3d=None):
        if self.training:
            preds_joints, pred_mano_results, gt_mano_results, preds_obj = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                                           mano_params=mano_params)
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj
        else:
            preds_joints, pred_mano_results, _, preds_obj = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                             roots3d=roots3d)
            return preds_joints, pred_mano_results, preds_obj