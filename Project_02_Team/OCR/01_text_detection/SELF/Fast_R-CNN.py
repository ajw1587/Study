# Fast R-CNN

class SlowROIPool(nn.Module):    
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.size = output_size
    
    # images : 원본 이미지
    # rois : region of interests
    # roi_idx : 
    def forward(self, images, rois, roi_idx):
        n = rois.shape[0] # region of interest의 수
        
        # 고정된 크기로 들어오기 때문에 전부 다 14x14
        h = images.size(2) # h : feature map height
        w = images.size(3) # w : feature map width
        
        # region of interst의 (x1, y1, x2, y2)의 행렬
        # 상대 좌표로 들어옴
        x1 = rois[:,0]
        y1 = rois[:,1]
        x2 = rois[:,2]
        y2 = rois[:,3]
        
        # region of interest의 상대좌표를 feature map에 맞게 절대좌표로 변환함
        x1 = np.floor(x1 * w).astype(int)
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)

        res = []
        
        # region of interest의 수만큼 순회
        for i in range(n):
            img = images[roi_idx[i]].unsqueeze(0) # roi_idx i번째 해당하는 feature map
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]] # 잘라내기
            img = self.maxpool(img) # adaptive average pooling
            res.append(img)
        res = torch.cat(res, dim=0)
        return res # 7x7x(# of region proposals)


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        rawnet = torchvision.models.vgg16_bn(pretrained=True)  # pre-trained된 vgg16_bn 모델 가져오기 
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1]) # 마지막 max pooling 제거
        self.roipool = SlowROIPool(output_size=(7, 7)) # 마지막 pooling layer, roi pooling으로 대체
        self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1])  # 마지막 fc layer 제거

        _x = Variable(torch.Tensor(1, 3, 224, 224))
        _r = np.array([[0., 0., 1., 1.]])
        _ri = np.array([0])
        _x = self.feature(self.roipool(self.seq(_x), _r, _ri).view(1, -1)) # 7x7x(# of region proposals)
        
        feature_dim = _x.size(1) 
        self.cls_score = nn.Linear(feature_dim, N_CLASS+1) # classifier
        self.bbox = nn.Linear(feature_dim, 4*(N_CLASS+1)) # bounding box regressor
        
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, inp, rois, ridx):
        res = inp # images
        res = self.seq(res) # ~pre-pooling
        res = self.roipool(res, rois, ridx) # roi pooling
        res = res.detach() # 연산 x
        res = res.view(res.size(0), -1)
        feat = self.feature(res) # fc layers

        cls_score = self.cls_score(feat) # classification result
        bbox = self.bbox(feat).view(-1, N_CLASS+1, 4) # bounding box regressor result
        
        return cls_score, bbox

    def calc_loss(self, probs, bbox, labels, gt_bbox):
        loss_sc = self.cel(probs, labels) # crossentropy loss
        
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4)
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)
        loss_loc = self.sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)
        
        # multi-task loss, crossentropy loss, smooth l1 loss
        lmb = 1.0
        loss = loss_sc + lmb * loss_loc
        
        return loss, loss_sc, loss_loc


def train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=False):
    sc, r_bbox = rcnn(img, rois, ridx) # class score, bbox
    loss, loss_sc, loss_loc = rcnn.calc_loss(sc, r_bbox, gt_cls, gt_tbbox) # losses
    fl = loss.data.cpu().numpy()[0]
    fl_sc = loss_sc.data.cpu().numpy()[0]
    fl_loc = loss_loc.data.cpu().numpy()[0]

    if not is_val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return fl, fl_sc, fl_loc


def train_epoch(run_set, is_val=False):
    I = 2   # number of image : 2
    B = 64  # number of rois per image : 64
    POS = int(B * 0.25)  # positive samples : 16
    NEG = B - POS # negative samples : 48
    
    # shffle images
    Nimg = len(run_set)
    perm = np.random.permutation(Nimg)
    perm = run_set[perm]
    
    losses = []
    losses_sc = []
    losses_loc = []
    
    # 전체 이미지를 I(=2)개씩만큼 처리
    for i in trange(0, Nimg, I):
        lb = i
        rb = min(i+I, Nimg)
        torch_seg = torch.from_numpy(perm[lb:rb]) # read 2 images
        img = Variable(train_imgs[torch_seg], volatile=is_val).cuda()
        ridx = []
        glo_ids = []

        for j in range(lb, rb):
            info = train_img_info[perm[j]]
            
            # roi의 positive, negative idx에 대한 리스트
            pos_idx = info['pos_idx']
            neg_idx = info['neg_idx']
            ids = []

            if len(pos_idx) > 0:
                ids.append(np.random.choice(pos_idx, size=POS))
            if len(neg_idx) > 0:
                ids.append(np.random.choice(neg_idx, size=NEG))
            if len(ids) == 0:
                continue
            ids = np.concatenate(ids, axis=0)
            
            # glo_ids : 두 이미지에 대한 positive, negative sample의 idx를 저장한 리스트
            glo_ids.append(ids)
            ridx += [j-lb] * ids.shape[0]

        if len(ridx) == 0:
            continue
        glo_ids = np.concatenate(glo_ids, axis=0)
        ridx = np.array(ridx)

        rois = train_roi[glo_ids]
        gt_cls = Variable(torch.from_numpy(train_cls[glo_ids]), volatile=is_val).cuda()
        gt_tbbox = Variable(torch.from_numpy(train_tbbox[glo_ids]), volatile=is_val).cuda()

        loss, loss_sc, loss_loc = train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)
        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)

    avg_loss = np.mean(losses)
    avg_loss_sc = np.mean(losses_sc)
    avg_loss_loc = np.mean(losses_loc)
    print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')
    
    return losses, losses_sc, losses_loc


def reg_to_bbox(img_size, reg, box):
    img_width, img_height = img_size
    bbox_width = box[:,2] - box[:,0] + 1.0
    bbox_height = box[:,3] - box[:,1] + 1.0
    bbox_ctr_x = box[:,0] + 0.5 * bbox_width
    bbox_ctr_y = box[:,1] + 0.5 * bbox_height

    bbox_width = bbox_width[:,np.newaxis]
    bbox_height = bbox_height[:,np.newaxis]
    bbox_ctr_x = bbox_ctr_x[:,np.newaxis]
    bbox_ctr_y = bbox_ctr_y[:,np.newaxis]

    out_ctr_x = reg[:,:,0] * bbox_width + bbox_ctr_x
    out_ctr_y = reg[:,:,1] * bbox_height + bbox_ctr_y

    out_width = bbox_width * np.exp(reg[:,:,2])
    out_height = bbox_height * np.exp(reg[:,:,3])

    return np.array([
        np.maximum(0, out_ctr_x - 0.5 * out_width),
        np.maximum(0, out_ctr_y - 0.5 * out_height),
        np.minimum(img_width, out_ctr_x + 0.5 * out_width),
        np.minimum(img_height, out_ctr_y + 0.5 * out_height)
    ]).transpose([1, 2, 0])