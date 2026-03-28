
# 利用Diffusion在MNIST手写数据集上实现手写数字生成

1. 配置环境：利用anaconda或miniconda创建虚拟环境MNISTDiffusion(名字自定义)，进入虚拟环境后安装依赖库
   1. 安装深度学习库：torch, torchvision
   https://pytorch.ac.cn/get-started/locally/
   2. 安装辅助作图库：matplotlib
   3. 安装进度条库：tqdm
2. 参考architecture.md实现UNet模型。或自定义模型结构。
3. 加载MNIST数据集。
   1. 从torchvision.datasets导入或下载MNIST数据集。
   2. 对数据集进行归一化处理。
   3. 创建数据加载器。
4. 结合Diffusion.py中的扩散模型类训练UNet模型，实现手写数字生成。
   1. 导入Diffusion模型类。
   2. 实例化Diffusion模型，推荐参数如下
   ```python
       diffusion = Diffusion(
        steps=1000,          # 扩散步数
        start=1e-4,          # beta起始值
        end=2e-2,            # beta结束值
        beta_type='cosine',  # 使用cosine调度
        s=0.008              # cosine调度的s参数
    )
   ```
   3. 设置训练的epoch, batch_size, optimizter, scheduler, loss_function等，推荐参数如下
      1. epoch=10
      2. batch_size=64(在创建数据加载器时设置)
      3. optimizter=torch.optim.Adam(diffusion.parameters(), lr=1e-4)
      4. scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizter, T_max=epoch)
      5. loss_function=torch.nn.MSELoss()
   4. 训练流程：
      1. 从数据加载器中获取一批数据$x$
      2. 选取随机加噪时间步$t$
      3. 利用Diffusion类的do_noise方法根据$t$和$x$获取输入$X$和噪声$Y$
      4. 将$X$输入UNet模型，得到预测$\hat{Y}$
      5. 根据$Y$和$\hat{Y}$，利用loss_function函数计算损失
      6. 训练UNet模型，更新参数
      ```python
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      ```
      7. 更新学习率
      ```python
        scheduler.step()
      ```
      8. 训练完一个epoch后保存模型参数
    5. 评估模型，利用Diffusion类的sample_all方法生成手写数字，利用matplotlib可视化生成结果
    ```python
        import matplotlib.pyplot as plt
        with torch.no_grad():
            samples = diffusion.sample_all(model, n=16, img_size=28)  # 生成16个样本
            samples = samples.cpu().numpy()

        # 显示样本
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            img = samples[i].transpose(1, 2, 0)  # 转换为HWC格式
            img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
            ax.imshow(img[:, :, 0], cmap='gray')  # 显示灰度图像
            ax.axis('off')
        plt.suptitle('Generated MNIST Samples')
        plt.savefig(f'result/generated_samples_mnist{epoch}.png')
        plt.close()
    ```
5. 可视化生成结果如example_output文件夹中的所示
![example_output](https://github.com/YouthGoblinBoy/MNIST-Diffusion/blob/main/example_output/generated_samples_mnist9.png)
6. 训练过程中的参考现象如下
   1. 在i7-7700K CPU上训练，每个epoch训练时间约3-4分钟
   2. 第一个epoch的平均损失在0.12左右，3epoch后的平均损失在0.04附近
7. 思考题，尝试回答以下问题，没有正确答案：
   1. 有一些生成出来的图是失败的，为什么？
   2. 为什么生成出来的大部分都是数字1？

