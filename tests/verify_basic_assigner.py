import sys, os, torch
sys.path.insert(0, r'D:/个人文件/文献/毕业论文参考/code/efficientteacher')
from models.assigner.yolo_anchor_assigner import YOLOAnchorAssigner
from models.assigner.scale_aware_assigner import ScaleAwareAssigner
A=[[[10,13],[16,30],[33,23]],[[30,61],[62,45],[59,119]],[[116,90],[156,198],[373,326]]]
S=[8,16,32]
def mp(B=2): return [torch.randn(B,3,80,80,86),torch.randn(B,3,40,40,86),torch.randn(B,3,20,20,86)]
def test():
 ba=YOLOAnchorAssigner(na=3,nl=3,anchors=A,anchor_t=4.0,stride=S,nc=80,ota=False)
 sa=ScaleAwareAssigner(nc=80,na=3,nl=3,anchors=A,anchor_t=4.0,stride=S,top_k=13)
 for s in range(200):
  torch.manual_seed(s);p=mp(2)
  nt=torch.randint(1,10,(1,)).item();tg=torch.zeros(nt,6)
  tg[:,0]=torch.randint(0,2,(nt,));tg[:,1]=torch.randint(0,80,(nt,))
  tg[:,2:4]=torch.rand(nt,2)*0.8+0.1;tg[:,4:6]=torch.rand(nt,2)*0.3+0.02
  tc,tb,ix,an=ba.build_targets(p,tg)
  bc=torch.zeros(nt,dtype=torch.int32)
  for li in range(3):
   b_i,_,gj_i,gi_i=ix[li]
   if b_i.shape[0]==0:continue
   for idx in range(b_i.shape[0]):
    img=b_i[idx].item()
    gx=(gi_i[idx].float()+tb[li][idx,0])/p[li].shape[3]
    gy=(gj_i[idx].float()+tb[li][idx,1])/p[li].shape[2]
    for gi_idx in range(nt):
     if tg[gi_idx,0].item()!=img:continue
     if abs(tg[gi_idx,2].item()-gx.item())<0.05 and abs(tg[gi_idx,3].item()-gy.item())<0.05:bc[gi_idx]+=1;break
  sc=torch.zeros(nt,dtype=torch.int32)
  for img in range(2):
   im=tg[:,0]==img;it=tg[im]
   if it.shape[0]==0:continue
   cnt=sa._count_basic_positives(p,it)
   for j,gi_idx in enumerate(torch.where(im)[0]):sc[gi_idx]=cnt[j]
  if not torch.equal(bc,sc):
   print('FAIL seed '+str(s)+': build='+str(bc.tolist())+' sa='+str(sc.tolist()));return False
 print('PASSED: 200 seeds identical');return True
test()