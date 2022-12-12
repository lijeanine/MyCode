# examples are in dice.py:UNetx Validation and Testing
# function of iou(Intersection of Union): iou = (A & B)/(A ∪ B) 
def score_perclass(vol_output, ground_truth, num_classes):
    dice_perclass = torch.zeros(num_classes)
    iou_perclass = torch.zeros(num_classes)
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
        iou_perclass[i] = torch.div(inter, union-inter)
    return dice_perclass, iou_perclass
