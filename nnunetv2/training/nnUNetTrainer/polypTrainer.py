from nnunetv2.training.nnUNetTrainer.polyp_testloader import test_dataset
from nnunetv2.paths import nnUNet_raw
from einops import *
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import *
from nnunetv2.evaluation.polyp.polyp_eval import *
import re
# import matplotlib
# matplotlib.use('tkAgg')

# import matplotlib.pyplot as plt


class PolypTrainer(nnUNetTrainer):
    base_ch = 16
    block = "FusedMBConv"  # ConvNeXtBlock  FusedMBConv
    use_my_unet = True
    network_name = "my_unet"
    project_prefix = "polyp"
    setting = 2
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug=True, job_id=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, debug, job_id)
        self.initial_lr = 1e-3
        self.backbone_lr = 1e-3  # 1e-4
        self.save_every = 10

        if self.debug:
            self.batch_size = 2
            self.num_iterations_per_epoch = 2
            self.num_val_iterations_per_epoch = 2
        else:
            pass

    def _get_deep_supervision_scales(self):
        pool_op_kernel_sizes = self.configuration_manager.pool_op_kernel_sizes
        # pool_op_kernel_sizes = pool_op_kernel_sizes[:-1]
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            pool_op_kernel_sizes), axis=0))
        # deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
        #     self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-2]
        if self.network.__class__.__name__ == "Res2Network":
            deep_supervision_scales[3] = deep_supervision_scales[2]
        deep_supervision_scales = deep_supervision_scales[:4]

        return deep_supervision_scales

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                  weight_ce=1, weight_dice=1.5,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        # comment by me
        # weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights_IXI so that they sum to 1
        weights = weights / weights.sum()

        # weights = np.array([1] * len(deep_supervision_scales))
        print(f"ds wegihts: {weights}")
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True
                                   ):

        network_configs = {"plans_manager": plans_manager,
                           "dataset_json": dataset_json,
                           "configuration_manager": configuration_manager,
                           "num_input_channels": num_input_channels,
                           "deep_supervision": enable_deep_supervision,
                           "base_ch": nnUNetTrainer.base_ch,
                           "block": nnUNetTrainer.block,
                           "use_my_unet": nnUNetTrainer.use_my_unet}

        return get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision,
                                      base_ch=PolypTrainer.base_ch,
                                      block=PolypTrainer.block,
                                      use_my_unet=PolypTrainer.use_my_unet,
                                      setting=PolypTrainer.setting), network_configs

    def configure_optimizers(self):
        backbone_params = list(map(id, self.network.backbone.parameters()))
        other_params = filter(lambda p: id(p) not in backbone_params, self.network.parameters())

        optimizer = torch.optim.SGD(
            [
                {'params': self.network.backbone.parameters(), 'lr': 1e-5},
                {'params': other_params}
            ],
            self.initial_lr, weight_decay=self.weight_decay,
            momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr,  self.backbone_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def run_training(self, dataset_id=None):
        self.on_train_start()
        if dataset_id is not None:
            self.project_prefix = f"Dataset{dataset_id}"
        if not self.debug and self.local_rank == 0:
            wandb.login(key="66b58ac7004a123a43487d7a6cf34ebb4571a7ea")
            self.initialize_wandb(project=f"{self.project_prefix}_{self.fold}",
                                  name=f"{self.network.__class__.__name__}_{self.job_id}_"
                                       f"{self.fold}_lr_{self.initial_lr}",
                                  dir=self.output_folder,
                                  id=None)
            print(f"debug: {self.debug}".center(50, "="))

        # self.num_iterations_per_epoch = 2
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            self.print_to_log_file(f"start training, {self.num_iterations_per_epoch}")

            if self.debug:
                for batch_id in range(1):
                    # print(f"batch_id: {batch_id}")
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
            else:
                print(f"num_iterations_per_epoch: {self.num_iterations_per_epoch}".center(50, "="))
                for batch_id in range(self.num_iterations_per_epoch):
                    # print(f"batch_id: {batch_id} / {250}===================\r", end="")
                    train_outputs.append(self.train_step(next(self.dataloader_train)))

            self.print_to_log_file(f"finished training")

            self.on_train_epoch_end(train_outputs)

            # self.real_validation_polyp()
            self.polyp_test()

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            torch.cuda.empty_cache()

        self.on_train_end()

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        # data_1 = rearrange(data, 'N C H W -> N H W C')
        # for i in range(data_1.shape[0]):
        #     plt.imshow(data_1[i]); plt.show()
        target = batch['target']
        # for t in target:
        #     print(np.unique(t.detach().cpu().numpy()))
        data = data.to(torch.float16).to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(torch.float16).to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(torch.float16).to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # print(f"data.shape: {data.shape}")
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    def real_validation_polyp(self):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                    use_mirroring=True,
                                    perform_everything_on_gpu=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)

        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = segmentation_export_pool._pool
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            tr_keys, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                tr_keys = tr_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            dataset_all = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys + val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]
                raise ValueError(f"next_stages should be None in 2d configuration.")

            results = []

            size_dict = load_json(
                join(os.path.abspath(
                os.path.join(self.preprocessed_dataset_folder, "..")), "size_dict.json"))

            keys = list(dataset_val.keys())
            if self.debug:
                keys = keys[:2]

            for k in list(keys):
                if self.debug and "CVC-300" not in k:
                    continue

                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2 * len(segmentation_export_pool._pool))
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2 * len(
                                                                   segmentation_export_pool._pool))

                data, seg, properties = dataset_all.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                val_data_folder, img_name = k.split("_")
                validation_output_folder_data = join(validation_output_folder, val_data_folder)

                maybe_mkdir_p(validation_output_folder_data)

                output_filename_truncated = join(validation_output_folder_data, img_name)

                start_time = end_time = -1

                try:
                    start_time = time()
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    end_time = time()
                except RuntimeError:
                    predictor.perform_everything_on_gpu = False
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    predictor.perform_everything_on_gpu = True

                prediction = F.interpolate(prediction, size=size_dict[k],
                                           mode='bilinear', align_corners=True)

                assert start_time != -1 and end_time != -1 and start_time < end_time
                # self.print_to_log_file(f"predicting {k} took {(end_time - start_time):.2f} s")

                prediction = prediction.cpu()

                # ref_file = join(self.preprocessed_dataset_folder_base, 'gt_segmentations', f"{k}.png")

                # this needs to go into background processes
                # print(output_filename_truncated)
                properties['shape_after_cropping_and_before_resampling'] = (1, *size_dict[k])
                properties['shape_before_cropping'] = (1, *size_dict[k])
                properties['bbox_used_for_cropping'] = [[0, 1],
                                                        [0, size_dict[k][0]],
                                                        [0, size_dict[k][1]]]
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, True),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                assert next_stages is None, "next stage should be none in 2d."

            _ = [r.get() for r in results]

        result_path = validation_output_folder
        pred_root = validation_output_folder

        gt_root = f"{nnUNet_raw}/polyp/TestDataset"

        datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

        if self.debug:
            tabs, results = evaluate_1(result_path, pred_root, gt_root,
                                       verbose=False, debug_dataset_len=1)  # only eval CVC-300
        else:
            tabs, results = evaluate_1(result_path, pred_root, gt_root,
                                       verbose=False, debug_dataset_len=len(datasets))  # only eval CVC-300
        if self.local_rank == 0:
            m_dscs = []
            m_ious = []
            m_maes = []
            for res in results:
                dataset_name = res[0]
                m_dsc = res[1]
                m_iou = res[2]
                mae = res[6]

                m_dscs.append(m_dsc)
                m_ious.append(m_iou)
                m_maes.append(mae)

                if not self.debug:
                    wandb.log(data={f"test/mdsc/{dataset_name}": m_dsc}, step=self.current_epoch)
                    wandb.log(data={f"test/miou/{dataset_name}": m_iou}, step=self.current_epoch)
                    wandb.log(data={f"test/mae/{dataset_name}": mae}, step=self.current_epoch)

            m_dscs = np.mean(m_dscs)
            m_ious = np.mean(m_ious)
            m_maes = np.mean(m_maes)

            self.print_to_log_file(f"meanDSC/all: {m_dscs:.3}")
            self.print_to_log_file(f"miou/all: {m_ious:.3}")
            self.print_to_log_file(f"m_maes/all: {m_maes:.3}")

            # save as (epoch, miou, dsc)
            if m_ious > self.best_score['miou'][1]:
                self.best_score['miou'] = (self.current_epoch, float(m_ious),
                                           float(m_dscs))
            if m_dscs > self.best_score['dsc'][2]:
                self.best_score['dsc'] = (self.current_epoch,
                                          float(m_ious), float(m_dscs))

            self.print_to_log_file(f"current best miou: {self.best_score['miou'][1]:.3f} "
                                   f"at epoch: {self.best_score['miou'][0]:.3f}, "
                                   f"{self.best_score['miou']}")

            self.print_to_log_file(f"current best dsc: {self.best_score['dsc'][2]:.3f} at epoch: "
                                   f"{self.best_score['dsc'][0]:.3f}, "
                                   f"{self.best_score['dsc']}")

            if m_dscs > self.best_metric:
                self.best_metric = m_dscs
                self.best_epoch = self.current_epoch
                self.save_checkpoint(join(self.output_folder, "dsc_slice_best.pth"))

        self.print_to_log_file(f"finished real validation")
        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()


    def polyp_test(self):
        test1path = '/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/polyp/TestDataset'
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        def test(model, path, dataset):
            data_path = os.path.join(path, dataset)
            image_root = '{}/images/'.format(data_path)
            gt_root = '{}/masks/'.format(data_path)
            model.eval()
            num1 = len(os.listdir(gt_root))
            test_loader = test_dataset(image_root, gt_root, 352)
            DSC = 0.0
            for i in range(num1):
                image, gt, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()

                # res, res1 = model(image)
                # res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)

                res = model(image)
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                # eval Dice

                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                res = F.softmax(res, dim=1).data.cpu().numpy().squeeze()
                res = res.argmax(0)

                input = res
                target = np.array(gt)
                N = gt.shape
                smooth = 1
                input_flat = np.reshape(input, (-1))
                target_flat = np.reshape(target, (-1))
                intersection = (input_flat * target_flat)
                dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
                dice = '{:.4f}'.format(dice)
                dice = float(dice)
                DSC = DSC + dice

            return DSC / num1

        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice = test(self.network, test1path, dataset)
            print('epoch: {}, dataset: {}, dice: {}'.format(self.current_epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)

            if not self.debug:
                wandb.log({f"dsc/{dataset}": dataset_dice}, step=self.current_epoch)

        self.set_deep_supervision_enabled(True)