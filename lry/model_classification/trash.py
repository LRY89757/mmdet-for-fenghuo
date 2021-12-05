Classes = ['GFI1-1', 'O622-8', 'GFI1-2', 'XGI2', 'E1-63', 'fake', 'O2500-4', 
'O9953', 'unrecognizable', 'O622-4', 'GFI2', 'others', '10GFEC', 'empty', 
'GFI2-R', 'O155-8', 'GFI1-3', 'O2500']
serial = list(range(len(Classes)))
# Classes = dict(zip(Classes, serial))
# print(Classes)
tranfer = ['GFI1-1', 'O622155-8', 'GFI12R2', 'XGI2', 'E1-63', 'fake', 'O2500622-4', 
'O9953GFEC', 'unrecognizable', 'O2500622-4', 'GFI12R2', 'others', 'O9953GFEC', 'empty', 
'GFI12R2', 'O622155-8', 'GFI1-3', 'O2500']

trans = dict(zip(Classes, tranfer))
print(trans)

Classes = ['GFI1-1', 'O2500622-4', 'others', 'O2500', 'O9953GFEC', 'unrecognizable', 'O622155-8', 'XGI2', 'GFI1-3', 'fake', 'GFI12R2', 'E1-63', 'empty']
serial = list(range(len(Classes)))
print(dict(zip(Classes, serial)))

# 保存模型 https://zhuanlan.zhihu.com/p/38056115
# torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
#                             'optimizer': optimizer.state_dict(),'alpha': loss.alpha, 'gamma': loss.gamma},
#                            checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')

Classes = list(set(tranfer))
print(Classes)
print(len(Classes))




