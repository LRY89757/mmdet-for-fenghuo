<div align="center">
  <img src="https://dian.org.cn/static/dian/images/logo.dd99a9865177.gif" width="600"/>



**Dian团队烽火智能运维项目GitHub仓库地址。**


（图像识别技术预研项目）


# 图像识别技术预研项目可行性研究报告



烽火通信公共研发部
2021年9月 
 
目录

目录	I
1. 技术开发概述	1
1.1. 市场需求分析	3
1.2. 行业情况和竞争对手情况	5
1.2.1. H公司动态	6
1.2.2. Z公司动态	6
1.3. 方案介绍	8
1.3.1. XXXXXXXXXXX	8
1.4. 主要技术	9
1.4.1. 图像的基本操作	9
1.4.2. 技术2	14
1.4.3. 技术3	14
1.4.4. 技术4	14
1.5. 研究进度、成果行驶及应用方向	14
1.5.1. XXXXXXXXXXX	14
2. 关键技术分析	15
2.1. 技术1	15
2.1.1. 技术说明	15
2.1.2. 技术难点	15
2.1.3. 解决方案	15
2.1.4. 风险分析	15
2.2. 技术1	15
2.2.1. 技术说明	15
2.2.2. 技术难点	15
2.2.3. 解决方案	15
2.2.4. 风险分析	15
2.3. 技术1	15
2.3.1. 技术说明	15
2.3.2. 技术难点	15
2.3.3. 解决方案	15
2.3.4. 风险分析	15
2.4. 技术1	16
2.4.1. 技术说明	16
2.4.2. 技术难点	16
2.4.3. 解决方案	16
2.4.4. 风险分析	16
2.5. 技术1	16
2.5.1. 技术说明	16
2.5.2. 技术难点	16
2.5.3. 解决方案	16
2.5.4. 风险分析	16
3. 公司现有技术状况	16
3.1. 人力资源	16
3.2. 技术积累	16
4. 知识产权分析	16
5. 经费概算及资源需求	16
6. 结论	17





 
 
1.	技术开发概述
	技术开发背景描述
本图像识别技术可追溯至1960年，最早的图像识别技术基于模式识别，被应用在办公自动化的领域 。随着深度学习的发展，图像识别技术依托于高质量的大规模数据集与不同的层次结构的神经网络，在学术界和工业界都取得了令人瞩目的成绩，图像技术在各个领域应用遍地开花。在电信运维领域图像识别技术也发挥着越来约重要的作用，具体到与我司业务相关的电信运维领域有如下场景：
1）机房巡检的应用
通过对机房进行图像采集，利用图片对目标设备进行检测和识别，判断机房是否存在安全隐患和异常状态的设备，如果存在则向机房运维人员及时告警和定位，将图像处理技术运用到信息通信机房代替人工巡视工作，对机房实施在线监控和优化，大大降低了巡检的时间。
2）在电信运维质检上的应用
随着电信行业的发展，持续增长的宽带业务需要更高的人工运维成本。在电信运维的装维质检中，需要人工识别施工现场图片以评估装维质量。传统的装维人工质检存在检测准确率低且人力成本高的问题。若采用图像识别技术，则可显著节约人力成本并提高准确率。具体到与我司业务相关产品有家庭网关、电视盒子与室外分线箱 类。在实际情况中，待质检图片会出现设备缺失、设备不清晰、设备错误等多种情况。如下图所示：
 
电信宽带业务量巨大 ，每天约产生万张上 述图片集，其中存在大量如图 (d) 和图 (e) 所示的图片。运维质检要求施工项目必须与图片 中待测物致，例如家庭网关安装项目中，图片中就必须出现家庭网关，若出现图 (d) 或图 (e) 就视为不合格。
3）无源设备管理上的应用
   由于ODN等无源设备，无法采用类似于有源设备网络管理系统的实现对设备资产、状态等的管理，因此对于无源设备的管理存在着巨大困难，通过图像识别无源ODN设备的序列号、设备型号、端口数量（包括已用和未用端口数量）等。实现无源设备和无源资产管理的自动化。
  
1.1.	市场需求分析
本项目为光网设备图像识别，主要解决三方面的市场需求：第一“辅助网络管理”，老设备网络单盘“自举”上网管的功能（以SDH、MSTP系列设备为例子）。第二“辅助运维”，通过识别单盘LED等的状态，向网络运维人员提供网络状态参考。第三“辅助资产管理”，通过图片识别设备中各个部件的二维码，以二维码信息为基础，网管系统可以构建起网络资产管理管理系统。
（1）“辅助网络管理”：在网络设备管理领域，老设备（SDH、DWDM、PTN、老OTN）等设备未开发单盘“自举”功能，需要通过手工根据单盘的所在的槽位进行配置，单盘配置错误将该设备或该单盘将无法实现与网管系统的通信，该设备或该单盘将无法正常工作。另外由于新设备都已支持的单盘“自举”功能，因此大部分的新的设备操作相关人员，缺乏单盘配置上管能力。因此本项目将实现、设备名称、单盘名称、单盘位置的设备的图像识别，为网管系统或网络操作人员提供设备单盘上管信息。
 
（2）“辅助运维”：在实际网络运行，出现部分网络故障，需要专业人士到现场查看网络设备运行状态，例如（2）某机盘脱管，与网管系统无法通信，需要专业工程师到现场查看该机盘的状态（如机盘完全不运行active等不亮，或者所有LED等都正常但无法上网管等）不同的状态有不同的网络故障定位方法。例如（2）网络机盘端口出现信号丢失告警，在光缆没有故障的情况下，需要到机房查看寻找故障定位原因，例如是否光纤连接，还是收发光模块故障或机盘故障等。 如上的这些需要专业工程到现场的工作，可以通过机房普通值守人员通过拍照，将其上传网络管理中心，来通过图像的自动识别，进行辅助运维工作。 
（3）“辅助资产管理”：传统的网络管控系统网络管理和控制非常强大，但是对于网络资产管理缺乏高效的方法。因此通过图像识别技术，识别每个单盘及子框的条形码，并结合第一部分“辅助上管”的信息，可以实现“精确”的基于整网、子网、行政区域、站点等多层次高效的资产管理功能。
 






1.2.	行业情况和竞争对手情况

1.2.1.	H公司动态
  友商H公司在图像识别技术领域有多年的积累，技术水平处于行业前列，并将图像识别技术做成云服务，其部分内容如下：
	人脸识别 ：基于深度学习的人脸识别方案，准确识别图片中的人脸信息，提供人脸属性识别、关键点定位、人脸1：1比对、人脸1：N识别、M:N识别、活体检测等能力。
	物体识别 ：准确识别图片中的物体类别、位置、置信度等综合信息。
	图像搜索 ：以图搜图，在指定图库中搜索出相同或相似的图片。
	人体分析：基于深度学习的人体识别方案，准确识别图像中的人体相关信息，提供人体检测与追踪、关键点定位、人流量统计、属性分析、人像分割、手势识别等能力。
	文字识别（OCR）： 识别图像中文字技术。
在电信网络运维领域，H公司利用图像技术实现机房智能巡检，并将其做成云服务提供给客户使用。在数据中心通过巡检通过机器人、摄像机等采集图片数据，结合AI技术和数据中心巡检经验，包括火点识别、开关、指示灯检测、AHU污垢识别等，支持图像分类、目标检测、图像分割等技术，自动判断机房异常状态，保障巡检质量，降低运维成本，实现无人巡检。

1.2.2.	Z公司动态
友商Z公司，积极布局机器视觉领域，目前国内机器视觉处于快速发展时期，企业间竞争激烈，处于百花齐放的阶段。据相关机构统计数据，中国机器视觉应用市场规模近几年一直保持快速增长，2019年已经达到14.56亿美元，3年内预计会达到200亿美元的规模。
机器视觉技术可广泛应用于工业、交通、医疗、教育、城市园区、商业、农林牧渔等各个领域。目前的机器视觉市场，工业领域占据了主要份额，是机器视觉的重要应用领域。在工业领域内，3C电子制造以及汽车制造行业又占据了大部分市场份额。
工业领域中，机器视觉可以在冶金、食品、电子制造、汽车、化工等各个领域应用，如生产工艺的检测、产品质量的检查、零部件的管理等。在3C电子制造行业，机器视觉在产品表面检测、触摸屏制造、AOI光学检测、电路板检测、产品装配等多个领域发挥重要作用。在汽车制造领域，主要应用于对汽车零部件的测量、检测，以及对车身、部件等对象的表面检测。
城市、园区领域，机器视觉可广泛用于城市建筑三维建模、人员的密集程度检测、人流分析、异常行为检测等场景。
商业领域，机器视觉可用于商品识别、数量检测、顾客喜好分析、支付、AR导览等场景。
交通领域，机器视觉可用于自动驾驶、车辆违章检测、车流密度检测、车牌识别、车辆排查、运行轨迹勾画、港口货物识别、机场车站闯入监测等多个场景。
医疗领域，机器视觉可用于病灶分析，协助医生进行诊断。
教育领域，机器视觉可提供沉浸式教学，也可用于工业场景中工艺流程的培训。
农林牧渔行业中，机器视觉可用来检测土地面积，判断动植物病理、生长情况，监测火灾等。
机器视觉市场蓬勃发展，对该领域的投资也不断增长。行业数据表明，2019年国内在机器视觉领域的投资融资超百亿。在获取更多资金后，机器视觉企业会加速发展，各种新技术也会加速落地，进入实际应用阶段。
Z公司作为5G设备的重要供应商之一，积极参与5G建设的同时，在机器视觉领域也逐步发力，在工业、园区等领域进行研究，和各行业龙头企业合作，共同推进5G场景下企业的智能化转型。在南京滨江5G智能制造基地，Z公司的机器视觉应用于产品质量检测，极大提升了检测效率。Z公司和大型铝材生产厂商神火集团开展合作，为神火集团园区提供5G+机器视觉解决方案，通过视觉分析方式，助力其检测生产过程，提取生产数据，保证生产的安全、稳定。
Z公司一直致力于机器视觉在工业领域的推广，在2020年的国际电信联盟第十六研究组ITU-T SG16全会上，由中国电信牵头，Z公司参与联合提出的“基于机器视觉的智能制造业务和应用需求”标准成功通过立项，Z公司担任联合编辑人。未来，Z公司将结合应用和需求场景并融合5G、MEC等技术，致力于机器视觉国际标准的研究和制定，推动产业形成统一格局，助力生态圈打造。

























1.3.	方案介绍
 

本项目总体方案围绕光网设备图像识别的场景，构建系统。主要实现光网辅助网络管理和辅助运维的功能，具体包括：
	设备位置及类型识别：
运用目标检测技术，检测图像中设备的位置及类别。并使用OCR文字识别技术识别设备丝印，辅助确定设备型号。
	单盘位置及类型识别。
基于目标检测提取出的设备位置，结合设备中单盘槽位的人工先验，裁剪提取出设备中每个盘位特征区域，进行图像分类判断单盘类别。并使用OCR文字识别技术识别单盘丝印，辅助确定单盘型号。
	单盘LED灯状态识别
基于提取出的单盘特征区域，裁剪出LED灯区域。根据LED灯RGB/HSI颜色特征识别判断LED灯状态。
 

1.4.	主要技术
本期预研项目所涉及光网络设备智能识别系统开发的主要技术如下：
1.4.1.	数据集标注
数据为人工智能提供动力，是机器学习模型建立的基础。每个AI应用程序都需要一个合适的数据集，它是创建高效、准确系统的重要组成部分，因此需要大量的优质的数据才能准确地执行复杂的任务。
由于该项目需要识别特定的光网络设备，网络上没有现成的数据集供使用。需要手动拍摄收集一定量机房内设备的图像数据，并使用相关数据标注软件进行标注，从而创建符合系统要求的数据集。

1.4.2.	目标检测技术
目标检测，也叫目标提取，是一种基于目标几何和统计特征的图像分割，它将目标的分割和识别合二为一。目标检测技术一般分为双阶段目标检测和单阶段目标检测两种方法，也可以根据是否使用锚框分为基于锚框和不基于锚框的两种方法。
双阶段目标检测首先通过选框检测器网络生成许多侯选框，对候选框进行挑选后从特征图上提取对应位置的特征，然后利用这些特征对侯选框进行微调，并预测类别；单阶段目标检测利用回归思想直接对输入图像生成侯选框及其对应的类别，相较于双阶段检测算法，具有计算量更小，推理速度更快的优点。基于锚框的目标检测方法利用预先设定好的锚框，为检测框的回归提供参考；基于无锚框的一些目标检测算法，通过回归中心点，回归边界框的角等方式来确定物体的位置。
在该项目中需要用到目标检测技术来检测设备与单盘的位置及类别。使用目标检测技术需要考虑和解决以下问题：
	多数原始拍摄图像存在一定角度倾斜，检测后需要进行实例分割裁剪出目标进行后续处理，但实例分割对设备边缘的识别不够完美。
	不同类别单盘外观差异较小，仅使用目标检测技术能否准确识别。
	目标检测往往计算时延较长，成为整个识别流程的性能瓶颈。

1.4.3.	OCR文字识别技术
OCR文字识别技术，是指针对印刷体字符，采用光学的方式将文字转换成为黑白点阵的图像文件，并通过识别技术将图像中的文字转换成文本格式，进行进一步编辑加工的技术。
在该项目中，初始使用目标检测技术大致判断设备和单盘的位置及类别。但由于单盘之间外形差异较小，还存在不同型号但外观相同的单盘，仅使用目标检测技术不足以确定单盘型号，因此还需辅助以OCR文字识别技术来识别各个单盘上的丝印，根据丝印文字来确定单盘的类别。使用OCR文字识别技术需要考虑和解决以下问题：
	图像质量不佳，亮度过暗时，OCR文字识别效果不佳。
	当目标检测与OCR文字识别的结果冲突时的优先次序。

1.4.4.	模型在线开发
深度学习开发环境的搭建所需工具软件较多，搭建过程也较为繁琐。光网络设备图像识别系统在进行模型训练的时候也需要搭建深度学习环境。因此，识别系统顺便提供模型在线开发环境及工具，将极大的方便模型开发人员开展工作，减少环境搭建的重复工作。同时，业界也有模型在线开发的实例，有较为成熟的开源组件提供支持，使得模型在线开发成为可能，且不用从零开始构建。

1.4.5.	训练可视化
深度神经网络就像一个黑盒子，其内部的组织、结构、以及其训练过程很难理清楚。通常涉及大量的调整、修改网络结构和尝试各种优化算法和超参数，相关技术通常是以经验性的成果作为基础。这给深度神经网络原理的理解和工程化带来了很大的挑战。所幸的是，数据可视化与生俱来的视觉特性可以弥补上述的部分缺陷，并描绘出更高层次的图像，在深度神经网络训练过程中助研究人员一臂之力。

1.4.6.	模型部署
光网络设备图像识别系统平台主要用于模型的开发与训练，一般部署在我司机房里，用户主要为我司模型开发人员。管控推理平台基于训练得到的模型使用现网数据进行推理预测，一般随网管部署在运营商机房，用户主要为运维人员。在两者网络不互通的情况下，难以采用在线的方式部署，模型只能采用离线的方式部署，即模型训练完成后需要以文件的格式导出供推理平台使用。








2.	关键技术分析
2.1.	光网络设备数据集标注
2.1.1.	技术说明
数据为人工智能提供动力，是机器学习模型建立的基础。每个AI应用程序都需要一个合适的数据集，它是创建高效、准确系统的重要组成部分，因此需要大量的优质的数据才能准确地执行复杂的任务。
由于该项目需要识别特定的光网络设备，网络上没有现成的数据集供使用。需要自行手动拍摄收集一定量机房内设备的图像数据，并使用相关图像标注软件进行标注，从而创建符合系统要求的数据集。
常用的数据集标注工具有Labelimg、Labelme等。以Labelimg工具为例，Labelimg是一款图片标记工具，通过在原始图像中标注目标物体位置并对每张图片生成相应的xml文件，可用作目标检测数据。
 

2.1.2.	技术难点
好的机器学习模型需要大量的优质的数据集，而数据集标注是一个繁琐的过程，手动标注数据集需要耗费大量的工作量。


2.1.3.	解决方案
 
方案1：借用检测器进行数据集标注
先用一个检测精度高的检测器把图片数据整体处理一遍，得到预标定的结果。然后借助标注工具对预标定的结果进行校正、删除或添加。该方案优点是当数据量很多的时候比较高效，可有效减少工作量。方案缺点为很难找到合适且强大的检测器。
方案2：人工数据集标注
采用人工手动的方式对数据集进行逐张标注，该方案优点是，较与检测器处理，手动标注的结果更加精确，缺点是工作量大。

2.1.4.	风险分析
目前光网络设备的图像数据较少，若不能进行补充或一些别的处理方法，得到的数据集质量难以保障，将会影响目标检测的精度。
2.2.	设备目标检测
2.2.1.	技术说明
光网络设备中由于单盘为细长型目标，对卷积神经网络进行特征提取存在一定挑战：高维的特征图内往往融入了相邻设备的特征表示，因此难以直接用目标检测器进行单盘设备型号的识别。
因此，该项目拟采用一种三阶段检测的流程：第一个阶段中，我们先对设备机箱进行目标检测，直接回归机箱的四个顶点，以应对输入图像的倾斜与透视形变；接着在第二个阶段中，我们提取出设备机箱槽位的感兴趣区域（region of interest, RoI），并依据设备机箱的人工先验直接对该图像区域进行处理，切分出每个单盘设备/空盘位的特征区域；最后在第三个阶段中，我们将每个单盘设备/空盘位的特征区域通过图像分类网络进行单盘型号/空盘位的识别。
 
2.2.2.	技术难点
	目标检测算法往往参数量、计算量较大，且计算时延较长，因此对于部署设备性能与推理时延有一定挑战；
	由于常规方式训练的检测器对于设备边缘无法实现细粒度的分割，则无法很好地从设备槽位中进一步提取单盘特征；
2.2.3.	解决方案
目前，学术界内基于无锚框（anchor-free）、单阶段（single-stage）、无非最大值抑制（non-maximum suppression）等技术的目标检测方法逐渐成熟，极大地提高了目标检测任务的效率，降低了推理时延，也因此对边缘部署设备更加友好。
由于需要对设备的边缘进行更细粒度的分割，一方面可以采用级联检测器（cascade detector），根据逐级递增的交并比（intersection over union, IoU）阈值作为标签映射（label assignment）的依据，起到提升预测结果的交并比的效果。另一方面可以采用针对交并比进行优化的损失函数代替原本针对坐标位置优化的损失函数，进一步提升定位的精度。
2.2.4.	风险分析
大多数输入的原始图像存在一定角度的倾斜与透视形变，并且清晰度较低，不利于单盘图像分类、OCR 文字识别等下游任务的效果。


2.3.	单盘丝印 OCR 文字识别
2.3.1.	技术说明
OCR文字识别技术，是指针对印刷体字符，采用光学的方式将文字转换成为黑白点阵的图像文件，并通过识别技术将图像中的文字转换成文本格式，进行进一步编辑加工的技术。
在该项目中，还需辅助以单盘丝印OCR文字识别技术来确定单盘的型号。单盘丝印识别根据对设备进行目标检测定位出的设备图像区域进行进一步的处理，依据丝印物理位置的人工先验进一步提取出丝印的图像区域，并进行OCR文字识别，从而识别出单盘设备的丝印文字。
2.3.2.	技术难点
机房照明情况不佳，拍摄图像存在模糊、过暗等质量不佳的情况，即人肉眼亦难以分辨，这对 OCR 文字识别的效果有一定挑战。
2.3.3.	解决方案
 
文字识别前先对图像文字区域进行预处理，通过高反差操作保留得到文字轮廓信息，然后与原图进行图层叠加，提高文字的精细程度，从而提高文字识别的精度。
2.3.4.	风险分析
设备机箱内往往有大量线缆等其他杂物，存在较多的丝印遮挡情况，此时则完全无法识别出丝印文字。


2.4.	模型在线开发
2.4.1.	技术说明
模型在线开发需要解决为在线用户提供机器学习和深度学习平台，供用户开发使用，模型开发测试完毕，模型能够顺利部署训练。重点关注有以下几点：
（1）快速生成模型开发作业所需的环境和计算资源；
（2）环境内置多个环境模块，支持主流成熟的机器学习框架和算法库；
（3）模型离线代码的上传、以及在线模型代码创建，支持模型文件的目录式管理；
（4）模型代码的一站式开发与调试（如：在线编辑、运行调试、结果查看、数据可视化等）；
2.4.2.	技术难点
（1）按用户定制在线开发环境，用户环境隔离。
（2）在线代码编辑与调试
（3）可视化数据
2.4.3.	解决方案
 
AI平台中模型开发应用提供模型在线开发的环境，Docker容器中cgorup实现硬件资源的配额，Docker容器中部署应用Python和Jupyter NoteBook两大核心组件；包含两大核心组件的Docker容器再集成TensorFlow、Keras、SKLearn、SparkMLlib中任意组合的组件，Docker的镜像模板，开发者选择指定的镜像模板创建镜像实例。
AI平台的模型开发前端在集成Docker中运行的Jupyter NoteBook服务的代码编辑，运行，调试等功能，提供开发者相互隔离的私有开发工作空间，Docker中根目录的链接到用户的私有工作目录，用户与用户之间相互隔离。训练模型脚本文件通过文件上传的方式从私有的容器中能上传的指定训练平台的地址。
AI平台中的在线开发环境能查看大数据平台数据集，从大数据平台读取数据到开发平台Docker容器中，供开发者训练模型使用。
2.4.4.	风险分析
（1）Docker容器部署在开发平台上，对开发平台资源调度是较大的挑战，特别是容器的部署和初始化，耗时较长，存在资源竞争问题；
（2）Jupyter Notebook非开源的软件系统，跨越的集成和网管扩展对前端来说是较大挑战。
（3）模型开发与模型训练系统之间有互操作，系统之间的资源共享和实效性是考验。

2.5.	训练可视化
2.5.1.	技术说明
深度神经网络就像一个黑盒子，其内部的组织、结构、以及其训练过程很难理清楚。通常涉及大量的调整、修改网络结构和尝试各种优化算法和超参数，相关技术通常是以经验性的成果作为基础。这给深度神经网络原理的理解和工程化带来了很大的挑战。所幸的是，数据可视化与生俱来的视觉特性可以弥补上述的部分缺陷，并描绘出更高层次的图像，在深度神经网络训练过程中助研究人员一臂之力。例如，在模型训练过程中，如果可以实时地绘制出梯度数据分布，就可快速检测并纠正消失梯度或爆炸梯度现象。
 
Tensorboard是tensorflow内置的一个可视化工具，它通过将TensorFlow程序输出的日志文件的信息可视化，使得tensorflow程序的理解、调试和优化更加简单高效。Tensorboard的可视化依赖于TensorFlow程序输出的日志文件，因而TensorFlow和tensorboard程序在不同的进程中运行。
Pytorch也提供了visdom工具，用于训练过程的可视化。Visdom是一个灵活的工具，用于创建、组织和共享实时丰富数据的可视化，包括数值、图像、文本、甚至是视频。用户可通过编程组织可视化空间，或通过用户接口为生成数据打造仪表板。
Tensorboard是所用深度学习框架中功能最为强大者，且不需要额外的编程组织可视化空间，仅需要通过一些操作将数据记录到文件中，然后其自动读取文件完成可视化。Pytorch框架的可视化工具功能远少于Tensorboard，且需要编程组织可视化空间。深度学习框架可视化工具基本都是具备训练进度及损失函数的查看。AI训练平台需要支持多种深度学习框架，同时需要支持模型训练任务的启停控制，支持可视化查看训练进度，以及实时查看训练详情。
2.5.2.	技术难点
训练可视化工具需要广泛的应用和长期的积累，形成良好的功能集合和展现方式，才能对算法优化、超参数调整起到帮助。训练可视化所涉及的技术难点主要如下：
（1）可视化工具功能定义。缺乏使用深度神经网络长期使用经验，难以定义需要的可视化功能。
（2）计算图、高维向量可视化等可视化展示。涉及自动绘图、动画生成等复杂的技术难点。当前网管仅有拓扑图、信号流图涉及绘图相关技术，与计算图、高维向量的WEB可视化展示相差较大。
2.5.3.	解决方案
通过集成TensorBoard实现模型训练可视化。TensorBoard从原理上就是读取模型训练输出的日志然后自动解析后组织可视化空间，生成可视化界面。可以通过编写一个库，用来仿照TensorBoard界面来记住可视化展示了。可以参考MXNet的可视化工具--MXBoard实现。通过TensorBoard logger库文件生成TensorBoard日志格式的文件。TensorBoard logger是TeamHG-Memex开源的一个库文件，可以以此为基础应用到其他的深度学习框架。
2.5.4.	风险分析
（1）需要理解TensorBoard日志格式，在需要的时候扩展tensorboard logger。
（2）可视化界面是TensorBoard，不属于自研产品。

2.6.	模型部署
2.6.1.	技术说明
光网络设备图像识别系统平台主要用于模型的开发与训练，一般部署在我司机房里，用户主要为我司模型开发人员。管控推理平台基于训练得到的模型使用现网数据进行推理预测，一般随网管部署在运营商机房，用户主要为运维人员。在两者网络不互通的情况下，难以采用在线的方式部署，模型只能采用离线的方式部署，即模型训练完成后需要以文件的格式导出供推理平台使用。
2.6.2.	技术难点
暂无
2.6.3.	解决方案
使用原始模型文件进行部署生成模型文件简单，且工作量小。每个深度学习框架均有自己的模型导出方法。例如Keras通过save方法将模型保存成h5文件，推理时通过load_model方法将二进制格式的文件转换成模型对象。Tensorflow.train.Saver.save或tf.saved_model.builder将模型保存成ckpt或者pb格式的模型文件。
2.6.4.	风险分析
（1）管控推理平台需要根据不同的模型采用不同的加载方法。
（2）管控推理平台需要准备深度学习环境。模型开发与推理如果不是同一人，推理平台难以知道需要准备什么样的深度学习环境。
（3）推理平台难以复用数据处理逻辑。


3.	公司现有技术状况
3.1.	人力资源
3.2.	技术积累

4.	知识产权分析

5.	经费概算及资源需求

6.	结论










（图像识别技术预研项目）


可行性研究报告







烽火通信公共研发部
2021年9月 
 
目录

目录	I
1. 技术开发概述	1
1.1. 市场需求分析	3
1.2. 行业情况和竞争对手情况	5
1.2.1. H公司动态	6
1.2.2. Z公司动态	6
1.3. 方案介绍	8
1.3.1. XXXXXXXXXXX	8
1.4. 主要技术	9
1.4.1. 图像的基本操作	9
1.4.2. 技术2	14
1.4.3. 技术3	14
1.4.4. 技术4	14
1.5. 研究进度、成果行驶及应用方向	14
1.5.1. XXXXXXXXXXX	14
2. 关键技术分析	15
2.1. 技术1	15
2.1.1. 技术说明	15
2.1.2. 技术难点	15
2.1.3. 解决方案	15
2.1.4. 风险分析	15
2.2. 技术1	15
2.2.1. 技术说明	15
2.2.2. 技术难点	15
2.2.3. 解决方案	15
2.2.4. 风险分析	15
2.3. 技术1	15
2.3.1. 技术说明	15
2.3.2. 技术难点	15
2.3.3. 解决方案	15
2.3.4. 风险分析	15
2.4. 技术1	16
2.4.1. 技术说明	16
2.4.2. 技术难点	16
2.4.3. 解决方案	16
2.4.4. 风险分析	16
2.5. 技术1	16
2.5.1. 技术说明	16
2.5.2. 技术难点	16
2.5.3. 解决方案	16
2.5.4. 风险分析	16
3. 公司现有技术状况	16
3.1. 人力资源	16
3.2. 技术积累	16
4. 知识产权分析	16
5. 经费概算及资源需求	16
6. 结论	17





 
 
1.	技术开发概述
	技术开发背景描述
本图像识别技术可追溯至1960年，最早的图像识别技术基于模式识别，被应用在办公自动化的领域 。随着深度学习的发展，图像识别技术依托于高质量的大规模数据集与不同的层次结构的神经网络，在学术界和工业界都取得了令人瞩目的成绩，图像技术在各个领域应用遍地开花。在电信运维领域图像识别技术也发挥着越来约重要的作用，具体到与我司业务相关的电信运维领域有如下场景：
1）机房巡检的应用
通过对机房进行图像采集，利用图片对目标设备进行检测和识别，判断机房是否存在安全隐患和异常状态的设备，如果存在则向机房运维人员及时告警和定位，将图像处理技术运用到信息通信机房代替人工巡视工作，对机房实施在线监控和优化，大大降低了巡检的时间。
2）在电信运维质检上的应用
随着电信行业的发展，持续增长的宽带业务需要更高的人工运维成本。在电信运维的装维质检中，需要人工识别施工现场图片以评估装维质量。传统的装维人工质检存在检测准确率低且人力成本高的问题。若采用图像识别技术，则可显著节约人力成本并提高准确率。具体到与我司业务相关产品有家庭网关、电视盒子与室外分线箱 类。在实际情况中，待质检图片会出现设备缺失、设备不清晰、设备错误等多种情况。如下图所示：
 
电信宽带业务量巨大 ，每天约产生万张上 述图片集，其中存在大量如图 (d) 和图 (e) 所示的图片。运维质检要求施工项目必须与图片 中待测物致，例如家庭网关安装项目中，图片中就必须出现家庭网关，若出现图 (d) 或图 (e) 就视为不合格。
3）无源设备管理上的应用
   由于ODN等无源设备，无法采用类似于有源设备网络管理系统的实现对设备资产、状态等的管理，因此对于无源设备的管理存在着巨大困难，通过图像识别无源ODN设备的序列号、设备型号、端口数量（包括已用和未用端口数量）等。实现无源设备和无源资产管理的自动化。
  
1.1.	市场需求分析
本项目为光网设备图像识别，主要解决三方面的市场需求：第一“辅助网络管理”，老设备网络单盘“自举”上网管的功能（以SDH、MSTP系列设备为例子）。第二“辅助运维”，通过识别单盘LED等的状态，向网络运维人员提供网络状态参考。第三“辅助资产管理”，通过图片识别设备中各个部件的二维码，以二维码信息为基础，网管系统可以构建起网络资产管理管理系统。
（1）“辅助网络管理”：在网络设备管理领域，老设备（SDH、DWDM、PTN、老OTN）等设备未开发单盘“自举”功能，需要通过手工根据单盘的所在的槽位进行配置，单盘配置错误将该设备或该单盘将无法实现与网管系统的通信，该设备或该单盘将无法正常工作。另外由于新设备都已支持的单盘“自举”功能，因此大部分的新的设备操作相关人员，缺乏单盘配置上管能力。因此本项目将实现、设备名称、单盘名称、单盘位置的设备的图像识别，为网管系统或网络操作人员提供设备单盘上管信息。
 
（2）“辅助运维”：在实际网络运行，出现部分网络故障，需要专业人士到现场查看网络设备运行状态，例如（2）某机盘脱管，与网管系统无法通信，需要专业工程师到现场查看该机盘的状态（如机盘完全不运行active等不亮，或者所有LED等都正常但无法上网管等）不同的状态有不同的网络故障定位方法。例如（2）网络机盘端口出现信号丢失告警，在光缆没有故障的情况下，需要到机房查看寻找故障定位原因，例如是否光纤连接，还是收发光模块故障或机盘故障等。 如上的这些需要专业工程到现场的工作，可以通过机房普通值守人员通过拍照，将其上传网络管理中心，来通过图像的自动识别，进行辅助运维工作。 
（3）“辅助资产管理”：传统的网络管控系统网络管理和控制非常强大，但是对于网络资产管理缺乏高效的方法。因此通过图像识别技术，识别每个单盘及子框的条形码，并结合第一部分“辅助上管”的信息，可以实现“精确”的基于整网、子网、行政区域、站点等多层次高效的资产管理功能。
 






1.2.	行业情况和竞争对手情况

1.2.1.	H公司动态
  友商H公司在图像识别技术领域有多年的积累，技术水平处于行业前列，并将图像识别技术做成云服务，其部分内容如下：
	人脸识别 ：基于深度学习的人脸识别方案，准确识别图片中的人脸信息，提供人脸属性识别、关键点定位、人脸1：1比对、人脸1：N识别、M:N识别、活体检测等能力。
	物体识别 ：准确识别图片中的物体类别、位置、置信度等综合信息。
	图像搜索 ：以图搜图，在指定图库中搜索出相同或相似的图片。
	人体分析：基于深度学习的人体识别方案，准确识别图像中的人体相关信息，提供人体检测与追踪、关键点定位、人流量统计、属性分析、人像分割、手势识别等能力。
	文字识别（OCR）： 识别图像中文字技术。
在电信网络运维领域，H公司利用图像技术实现机房智能巡检，并将其做成云服务提供给客户使用。在数据中心通过巡检通过机器人、摄像机等采集图片数据，结合AI技术和数据中心巡检经验，包括火点识别、开关、指示灯检测、AHU污垢识别等，支持图像分类、目标检测、图像分割等技术，自动判断机房异常状态，保障巡检质量，降低运维成本，实现无人巡检。

1.2.2.	Z公司动态
友商Z公司，积极布局机器视觉领域，目前国内机器视觉处于快速发展时期，企业间竞争激烈，处于百花齐放的阶段。据相关机构统计数据，中国机器视觉应用市场规模近几年一直保持快速增长，2019年已经达到14.56亿美元，3年内预计会达到200亿美元的规模。
机器视觉技术可广泛应用于工业、交通、医疗、教育、城市园区、商业、农林牧渔等各个领域。目前的机器视觉市场，工业领域占据了主要份额，是机器视觉的重要应用领域。在工业领域内，3C电子制造以及汽车制造行业又占据了大部分市场份额。
工业领域中，机器视觉可以在冶金、食品、电子制造、汽车、化工等各个领域应用，如生产工艺的检测、产品质量的检查、零部件的管理等。在3C电子制造行业，机器视觉在产品表面检测、触摸屏制造、AOI光学检测、电路板检测、产品装配等多个领域发挥重要作用。在汽车制造领域，主要应用于对汽车零部件的测量、检测，以及对车身、部件等对象的表面检测。
城市、园区领域，机器视觉可广泛用于城市建筑三维建模、人员的密集程度检测、人流分析、异常行为检测等场景。
商业领域，机器视觉可用于商品识别、数量检测、顾客喜好分析、支付、AR导览等场景。
交通领域，机器视觉可用于自动驾驶、车辆违章检测、车流密度检测、车牌识别、车辆排查、运行轨迹勾画、港口货物识别、机场车站闯入监测等多个场景。
医疗领域，机器视觉可用于病灶分析，协助医生进行诊断。
教育领域，机器视觉可提供沉浸式教学，也可用于工业场景中工艺流程的培训。
农林牧渔行业中，机器视觉可用来检测土地面积，判断动植物病理、生长情况，监测火灾等。
机器视觉市场蓬勃发展，对该领域的投资也不断增长。行业数据表明，2019年国内在机器视觉领域的投资融资超百亿。在获取更多资金后，机器视觉企业会加速发展，各种新技术也会加速落地，进入实际应用阶段。
Z公司作为5G设备的重要供应商之一，积极参与5G建设的同时，在机器视觉领域也逐步发力，在工业、园区等领域进行研究，和各行业龙头企业合作，共同推进5G场景下企业的智能化转型。在南京滨江5G智能制造基地，Z公司的机器视觉应用于产品质量检测，极大提升了检测效率。Z公司和大型铝材生产厂商神火集团开展合作，为神火集团园区提供5G+机器视觉解决方案，通过视觉分析方式，助力其检测生产过程，提取生产数据，保证生产的安全、稳定。
Z公司一直致力于机器视觉在工业领域的推广，在2020年的国际电信联盟第十六研究组ITU-T SG16全会上，由中国电信牵头，Z公司参与联合提出的“基于机器视觉的智能制造业务和应用需求”标准成功通过立项，Z公司担任联合编辑人。未来，Z公司将结合应用和需求场景并融合5G、MEC等技术，致力于机器视觉国际标准的研究和制定，推动产业形成统一格局，助力生态圈打造。

























1.3.	方案介绍
 

本项目总体方案围绕ODN接头盒智能识别的场景，构建系统。主要实现以下功能：
	接头盒光缆口类型识别：
运用目标检测技术，检测接头盒图像中光缆口的位置。裁剪光缆口特征区域进行图像分类，对光缆口类型进行判断，为入口或出口。
	接头盒光缆口连接状态识别。
基于目标检测提取出的光缆口位置，裁剪提取出光缆口的特征区域，进行图像分类判断光缆口的连接状态，未连接或已连接。
	统计未连接光缆口数
基于检测出来的光缆口连接状态，分别统计未连接出口数和未连接入口数。
 

1.4.	主要技术
本期预研项目所涉及ODN接口盒智能识别系统开发的主要技术如下：
1.4.1.	数据集标注
数据为人工智能提供动力，是机器学习模型建立的基础。每个AI应用程序都需要一个合适的数据集，它是创建高效、准确系统的重要组成部分，因此需要大量的优质的数据才能准确地执行复杂的任务。
由于该项目需要识别特定的光网络设备，网络上没有现成的数据集供使用。需要手动拍摄收集一定量机房内设备的图像数据，并使用相关图像标注软件进行标注，从而创建符合系统要求的数据集。

1.4.2.	目标检测技术
目标检测，也叫目标提取，是一种基于目标几何和统计特征的图像分割，它将目标的分割和识别合二为一。目标检测技术一般分为双阶段目标检测和单阶段目标检测两种方法，也可以根据是否使用锚框分为基于锚框和不基于锚框的两种方法。
双阶段目标检测首先通过选框检测器网络生成许多侯选框，对候选框进行挑选后从特征图上提取对应位置的特征，然后利用这些特征对侯选框进行微调，并预测类别；单阶段目标检测利用回归思想直接对输入图像生成侯选框及其对应的类别，相较于双阶段检测算法，具有计算量更小，推理速度更快的优点。基于锚框的目标检测方法利用预先设定好的锚框，为检测框的回归提供参考；基于无锚框的一些目标检测算法，通过回归中心点，回归边界框的角等方式来确定物体的位置。
在该项目中需要用到目标检测技术来检测ODN接口盒光缆口的位置、类型以及状态。使用目标检测技术需要考虑和解决以下问题：
	多数原始拍摄图像存在一定角度倾斜，检测后需要进行实例分割裁剪出目标进行后续处理，但实例分割对设备边缘的识别不够完美。
	目标检测往往计算时延较长，成为整个识别流程的性能瓶颈。

1.4.3.	模型在线开发
深度学习开发环境的搭建所需工具软件较多，搭建过程也较为繁琐。ODN接口盒智能识别系统在进行模型训练的时候也需要搭建深度学习环境。因此，识别系统顺便提供模型在线开发环境及工具，将极大的方便模型开发人员开展工作，减少环境搭建的重复工作。同时，业界也有模型在线开发的实例，有较为成熟的开源组件提供支持，使得模型在线开发成为可能，且不用从零开始构建。

1.4.4.	训练可视化
深度神经网络就像一个黑盒子，其内部的组织、结构、以及其训练过程很难理清楚。通常涉及大量的调整、修改网络结构和尝试各种优化算法和超参数，相关技术通常是以经验性的成果作为基础。这给深度神经网络原理的理解和工程化带来了很大的挑战。所幸的是，数据可视化与生俱来的视觉特性可以弥补上述的部分缺陷，并描绘出更高层次的图像，在深度神经网络训练过程中助研究人员一臂之力。
1.4.5.	模型部署
接口盒智能识别系统平台主要用于模型的开发与训练，一般部署在我司机房里，用户主要为我司模型开发人员。管控推理平台基于训练得到的模型使用现网数据进行推理预测，一般随网管部署在运营商机房，用户主要为运维人员。在两者网络不互通的情况下，难以采用在线的方式部署，模型只能采用离线的方式部署，即模型训练完成后需要以文件的格式导出供推理平台使用。

2.	关键技术分析
2.1.	ODN接口盒数据集标注
2.1.1.	技术说明
数据为人工智能提供动力，是机器学习模型建立的基础。每个AI应用程序都需要一个合适的数据集，它是创建高效、准确系统的重要组成部分，因此需要大量的优质的数据才能准确地执行复杂的任务。
由于该项目需要识别特定的接口盒设备，网络上没有现成的数据集供使用。需要自行手动拍摄收集一定量机房内设备的图像数据，并使用相关图像标注软件进行标注，从而创建符合系统要求的数据集。
常用的数据集标注工具有Labelimg、Labelme等。以Labelimg工具为例，Labelimg是一款图片标记工具，通过在原始图像中标注目标物体位置并对每张图片生成相应的xml文件，可用作目标检测数据。
 
2.1.2.	技术难点
好的机器学习模型需要大量的优质的数据集，而数据集标注是一个繁琐的过程，手动标注数据集需要耗费大量的工作量。
2.1.3.	解决方案
 
方案1：借用检测器进行数据集标注
先用一个检测精度高的检测器把图片数据整体处理一遍，得到预标定的结果。然后借助标注工具对预标定的结果进行校正、删除或添加。该方案优点是当数据量很多的时候比较高效，可有效减少工作量。方案缺点为很难找到合适且强大的检测器。
方案2：人工数据集标注
采用人工的方式对数据集进行逐张标注，该方案优点是，较与检测器处理，手动标注的结果更加精确，缺点是工作量大。
2.1.4.	风险分析
目前光网络设备的图像较少，若不能进行补充或一些别的处理方法，得到的数据集质量难以保障，将会影响目标检测的精度。

2.2.	接口盒目标检测
2.2.1.	技术说明
接口盒设备中由于光缆口为细小型目标，对卷积神经网络进行特征提取存在一定挑战：高维的特征图内往往融入了相邻设备的特征表示，因此难以直接用目标检测器直接进行光缆口状态以及类型的识别。
因此，该项目拟采用了一种双阶段检测的流程：第一个阶段中，我们先对接口盒设备进行目标检测，回归光缆口的四个顶点，以应对输入图像的倾斜与透视形变，提取出设备中各个光缆口的感兴趣区域（region of interest, RoI），切分出每个光缆口的特征区域；然后在第二个阶段，我们对每个光缆口的特征区域通过图像分类网络进行类型以及连接状态的识别。
 
2.2.2.	技术难点
	目标检测算法往往参数量、计算量较大，且计算时延较长；对于部署设备性能与推理时延有一定挑战
	由于常规方式训练的检测器对于接口边缘无法实现细粒度的分割，则无法很好地从中设备中进一步提取接口特征。
2.2.3.	解决方案
目前，学术界内基于无锚框（anchor-free）、单阶段（single-stage）、无非最大值抑制（non-maximum suppression）等技术的目标检测方法逐渐成熟，极大地提高了目标检测任务的效率，降低了推理时延，也因此对边缘部署设备更加友好。
由于需要对接口的边缘进行更细粒度的分割，一方面可以采用级联检测器（cascade detector），根据逐级递增的交并比（intersection over union, IoU）阈值作为标签映射（label assignment）的依据，起到提升预测结果的交并比的效果；另一方面可以采用针对交并比进行优化的损失函数代替原本针对坐标位置优化的损失函数，进一步提升定位的精度。
2.2.4.	风险分析
大多数输入的原始图像存在一定角度的倾斜与透视形变，并且清晰度较低，会影响目标检测的效果。


2.3.	模型在线开发
2.3.1.	技术说明
模型在线开发需要解决为在线用户提供机器学习和深度学习平台，供用户开发使用，模型开发测试完毕，模型能够顺利部署训练。重点关注有以下几点：
（1）快速生成模型开发作业所需的环境和计算资源；
（2）环境内置多个环境模块，支持主流成熟的机器学习框架和算法库；
（3）模型离线代码的上传、以及在线模型代码创建，支持模型文件的目录式管理；
（4）模型代码的一站式开发与调试（如：在线编辑、运行调试、结果查看、数据可视化等）；
2.3.2.	技术难点
（1）按用户定制在线开发环境，用户环境隔离。
（2）在线代码编辑与调试
（3）可视化数据
2.3.3.	解决方案
 
AI平台中模型开发应用提供模型在线开发的环境，Docker容器中cgorup实现硬件资源的配额，Docker容器中部署应用Python和Jupyter NoteBook两大核心组件；包含两大核心组件的Docker容器再集成TensorFlow、Keras、SKLearn、SparkMLlib中任意组合的组件，Docker的镜像模板，开发者选择指定的镜像模板创建镜像实例。
AI平台的模型开发前端在集成Docker中运行的Jupyter NoteBook服务的代码编辑，运行，调试等功能，提供开发者相互隔离的私有开发工作空间，Docker中根目录的链接到用户的私有工作目录，用户与用户之间相互隔离。训练模型脚本文件通过文件上传的方式从私有的容器中能上传的指定训练平台的地址。
AI平台中的在线开发环境能查看大数据平台数据集，从大数据平台读取数据到开发平台Docker容器中，供开发者训练模型使用。
2.3.4.	风险分析
（1）Docker容器部署在开发平台上，对开发平台资源调度是较大的挑战，特别是容器的部署和初始化，耗时较长，存在资源竞争问题；
（2）Jupyter Notebook非开源的软件系统，跨越的集成和网管扩展对前端来说是较大挑战。
（3）模型开发与模型训练系统之间有互操作，系统之间的资源共享和实效性是考验。
2.4.	训练可视化
2.4.1.	技术说明
深度神经网络就像一个黑盒子，其内部的组织、结构、以及其训练过程很难理清楚。通常涉及大量的调整、修改网络结构和尝试各种优化算法和超参数，相关技术通常是以经验性的成果作为基础。这给深度神经网络原理的理解和工程化带来了很大的挑战。所幸的是，数据可视化与生俱来的视觉特性可以弥补上述的部分缺陷，并描绘出更高层次的图像，在深度神经网络训练过程中助研究人员一臂之力。例如，在模型训练过程中，如果可以实时地绘制出梯度数据分布，就可快速检测并纠正消失梯度或爆炸梯度现象。
 
Tensorboard是tensorflow内置的一个可视化工具，它通过将TensorFlow程序输出的日志文件的信息可视化，使得tensorflow程序的理解、调试和优化更加简单高效。Tensorboard的可视化依赖于TensorFlow程序输出的日志文件，因而TensorFlow和tensorboard程序在不同的进程中运行。
Pytorch也提供了visdom工具，用于训练过程的可视化。Visdom是一个灵活的工具，用于创建、组织和共享实时丰富数据的可视化，包括数值、图像、文本、甚至是视频。用户可通过编程组织可视化空间，或通过用户接口为生成数据打造仪表板。
Tensorboard是所用深度学习框架中功能最为强大者，且不需要额外的编程组织可视化空间，仅需要通过一些操作将数据记录到文件中，然后其自动读取文件完成可视化。Pytorch框架的可视化工具功能远少于Tensorboard，且需要编程组织可视化空间。深度学习框架可视化工具基本都是具备训练进度及损失函数的查看。AI训练平台需要支持多种深度学习框架，同时需要支持模型训练任务的启停控制，支持可视化查看训练进度，以及实时查看训练详情。
2.4.2.	技术难点
训练可视化工具需要广泛的应用和长期的积累，形成良好的功能集合和展现方式，才能对算法优化、超参数调整起到帮助。训练可视化所涉及的技术难点主要如下：
（1）可视化工具功能定义。缺乏使用深度神经网络长期使用经验，难以定义需要的可视化功能。
（2）计算图、高维向量可视化等可视化展示。涉及自动绘图、动画生成等复杂的技术难点。当前网管仅有拓扑图、信号流图涉及绘图相关技术，与计算图、高维向量的WEB可视化展示相差较大。
2.4.3.	解决方案
通过集成TensorBoard实现模型训练可视化。TensorBoard从原理上就是读取模型训练输出的日志然后自动解析后组织可视化空间，生成可视化界面。可以通过编写一个库，用来仿照TensorBoard界面来记住可视化展示了。可以参考MXNet的可视化工具--MXBoard实现。通过TensorBoard logger库文件生成TensorBoard日志格式的文件。TensorBoard logger是TeamHG-Memex开源的一个库文件，可以以此为基础应用到其他的深度学习框架。
2.4.4.	风险分析
（1）需要理解TensorBoard日志格式，在需要的时候扩展tensorboard logger。
（2）可视化界面是TensorBoard，不属于自研产品。

2.5.	模型部署
2.5.1.	技术说明
ODR接口盒智能识别系统平台主要用于模型的开发与训练，一般部署在我司机房里，用户主要为我司模型开发人员。管控推理平台基于训练得到的模型使用现网数据进行推理预测，一般随网管部署在运营商机房，用户主要为运维人员。在两者网络不互通的情况下，难以采用在线的方式部署，模型只能采用离线的方式部署，即模型训练完成后需要以文件的格式导出供推理平台使用。
2.5.2.	技术难点
暂无
2.5.3.	解决方案
使用原始模型文件进行部署生成模型文件简单，且工作量小。每个深度学习框架均有自己的模型导出方法。例如Keras通过save方法将模型保存成h5文件，推理时通过load_model方法将二进制格式的文件转换成模型对象。Tensorflow.train.Saver.save或tf.saved_model.builder将模型保存成ckpt或者pb格式的模型文件。
2.5.4.	风险分析
（1）管控推理平台需要根据不同的模型采用不同的加载方法。
（2）管控推理平台需要准备深度学习环境。模型开发与推理如果不是同一人，推理平台难以知道需要准备什么样的深度学习环境。
（3）推理平台难以复用数据处理逻辑。


3.	公司现有技术状况
3.1.	人力资源
3.2.	技术积累

4.	知识产权分析

5.	经费概算及资源需求

6.	结论



















<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>


[![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)


  <img src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png"/>


[📘Documentation](https://mmdetection.readthedocs.io/en/v2.18.1/) |
[🛠️Installation](https://mmdetection.readthedocs.io/en/v2.18.1/get_started.html) |
[👀Model Zoo](https://mmdetection.readthedocs.io/zh_CN/v2.18.1/model_zoo.html) |
[🆕Update News](https://mmdetection.readthedocs.io/en/v2.18.1/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection/issues/new/choose)

</div>

## Introduction

English | [简体中文](README_zh-CN.md)

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.3+**.

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.
</details>


Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

**2.18.1** was released in 15/11/2021:
- Release [QueryInst](http://arxiv.org/abs/2105.01928) pre-trained weights.
- Support plot confusion matrix.
- Fix SpatialReductionAttention in PVT and fix trunc_normal_init in both PVT and Swin-Transformer.

Please refer to [changelog.md](docs/changelog.md) for details and release history.

For compatibility changes between different versions of MMDetection, please refer to [compatibility.md](docs/compatibility.md).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).


<details open>
<summary>Supported backbones:</summary>

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] VGG (ICLR'2015)
- [x] MobileNetV2 (CVPR'2018)
- [x] HRNet (CVPR'2019)
- [x] RegNet (CVPR'2020)
- [x] Res2Net (TPAMI'2020)
- [x] ResNeSt (ArXiv'2020)
- [X] Swin (CVPR'2021)
- [x] PVT (ICCV'2021)
- [x] PVTv2 (ArXiv'2021)
</details>

<details open>
<summary>Supported methods:</summary>

- [x] [RPN (NeurIPS'2015)](configs/rpn)
- [x] [Fast R-CNN (ICCV'2015)](configs/fast_rcnn)
- [x] [Faster R-CNN (NeurIPS'2015)](configs/faster_rcnn)
- [x] [Mask R-CNN (ICCV'2017)](configs/mask_rcnn)
- [x] [Cascade R-CNN (CVPR'2018)](configs/cascade_rcnn)
- [x] [Cascade Mask R-CNN (CVPR'2018)](configs/cascade_rcnn)
- [x] [SSD (ECCV'2016)](configs/ssd)
- [x] [RetinaNet (ICCV'2017)](configs/retinanet)
- [x] [GHM (AAAI'2019)](configs/ghm)
- [x] [Mask Scoring R-CNN (CVPR'2019)](configs/ms_rcnn)
- [x] [Double-Head R-CNN (CVPR'2020)](configs/double_heads)
- [x] [Hybrid Task Cascade (CVPR'2019)](configs/htc)
- [x] [Libra R-CNN (CVPR'2019)](configs/libra_rcnn)
- [x] [Guided Anchoring (CVPR'2019)](configs/guided_anchoring)
- [x] [FCOS (ICCV'2019)](configs/fcos)
- [x] [RepPoints (ICCV'2019)](configs/reppoints)
- [x] [Foveabox (TIP'2020)](configs/foveabox)
- [x] [FreeAnchor (NeurIPS'2019)](configs/free_anchor)
- [x] [NAS-FPN (CVPR'2019)](configs/nas_fpn)
- [x] [ATSS (CVPR'2020)](configs/atss)
- [x] [FSAF (CVPR'2019)](configs/fsaf)
- [x] [PAFPN (CVPR'2018)](configs/pafpn)
- [x] [Dynamic R-CNN (ECCV'2020)](configs/dynamic_rcnn)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CARAFE (ICCV'2019)](configs/carafe/README.md)
- [x] [DCNv2 (CVPR'2019)](configs/dcn/README.md)
- [x] [Group Normalization (ECCV'2018)](configs/gn/README.md)
- [x] [Weight Standardization (ArXiv'2019)](configs/gn+ws/README.md)
- [x] [OHEM (CVPR'2016)](configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py)
- [x] [Soft-NMS (ICCV'2017)](configs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py)
- [x] [Generalized Attention (ICCV'2019)](configs/empirical_attention/README.md)
- [x] [GCNet (ICCVW'2019)](configs/gcnet/README.md)
- [x] [Mixed Precision (FP16) Training (ArXiv'2017)](configs/fp16/README.md)
- [x] [InstaBoost (ICCV'2019)](configs/instaboost/README.md)
- [x] [GRoIE (ICPR'2020)](configs/groie/README.md)
- [x] [DetectoRS (ArXiv'2020)](configs/detectors/README.md)
- [x] [Generalized Focal Loss (NeurIPS'2020)](configs/gfl/README.md)
- [x] [CornerNet (ECCV'2018)](configs/cornernet/README.md)
- [x] [Side-Aware Boundary Localization (ECCV'2020)](configs/sabl/README.md)
- [x] [YOLOv3 (ArXiv'2018)](configs/yolo/README.md)
- [x] [PAA (ECCV'2020)](configs/paa/README.md)
- [x] [YOLACT (ICCV'2019)](configs/yolact/README.md)
- [x] [CentripetalNet (CVPR'2020)](configs/centripetalnet/README.md)
- [x] [VFNet (ArXiv'2020)](configs/vfnet/README.md)
- [x] [DETR (ECCV'2020)](configs/detr/README.md)
- [x] [Deformable DETR (ICLR'2021)](configs/deformable_detr/README.md)
- [x] [CascadeRPN (NeurIPS'2019)](configs/cascade_rpn/README.md)
- [x] [SCNet (AAAI'2021)](configs/scnet/README.md)
- [x] [AutoAssign (ArXiv'2020)](configs/autoassign/README.md)
- [x] [YOLOF (CVPR'2021)](configs/yolof/README.md)
- [x] [Seasaw Loss (CVPR'2021)](configs/seesaw_loss/README.md)
- [x] [CenterNet (CVPR'2019)](configs/centernet/README.md)
- [x] [YOLOX (ArXiv'2021)](configs/yolox/README.md)
- [x] [SOLO (ECCV'2020)](configs/solo/README.md)
- [x] [QueryInst (ICCV'2021)](configs/queryinst/README.md)
</details>

Some other methods are also supported in [projects using MMDetection](./docs/projects.md).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with new dataset](docs/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/customize_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [useful tools](docs/useful_tools.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection. Ongoing projects can be found in out [GitHub Projects](https://github.com/open-mmlab/mmdetection/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
