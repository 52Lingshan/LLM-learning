# MIT 免费 AI 教材清单

> 来源：cyrilXBT 整理的 AI 教材清单  
> 整理日期：2026-05-11

## 一句话总结

这是一份适合系统学习 AI / ML 的免费教材路线图，覆盖机器学习基础、数学基础、深度学习、强化学习、计算机视觉、机器学习系统、生产部署和 AI 公平性等方向。

## 适合人群

适合：

- 想系统搞懂 AI / ML 原理的人
- 有一定数学和编程基础的人
- 想从科普阅读进入严肃学习的人
- 做 LLM、Agent、模型训练、推理系统、MLOps 的工程师

不太适合：

- 只是想泛泛了解 AI 概念的人
- 暂时不想啃数学、公式和代码的人

如果只是想“了解 AI”，这些书偏硬，科普书或入门视频更合适。若想真正理解 AI 是怎么回事，这些教材比碎片化看 100 篇科普文章更有价值。

---

## 教材清单

### 一、基础

#### 1. Foundations of Machine Learning

- 作者：Mehryar Mohri 等
- 链接：<https://cs.nyu.edu/~mohri/mlbook/>
- 关键词：机器学习理论、PAC 学习、泛化、核方法
- 定位：经典机器学习理论教材

适合用来建立 ML 的理论骨架，尤其是理解泛化能力、学习理论和经典算法基础。

#### 2. Mathematics for Machine Learning

- 链接：<https://mml-book.github.io/>
- 关键词：线性代数、概率、微积分、优化
- 定位：机器学习数学基础教材

适合作为补数学的主线资料。对于理解深度学习、概率模型、优化算法都很重要。

#### 3. Probabilistic Machine Learning Part 2

- 作者：Kevin Murphy
- 链接：<https://probml.github.io/pml-book/book2.html>
- 关键词：概率机器学习、高级概率模型、贝叶斯方法
- 定位：概率视角下的高级机器学习教材

适合已经有 ML 基础后，进一步从概率建模角度理解机器学习。

---

### 二、深度学习

#### 4. Deep Learning

- 作者：Ian Goodfellow、Yoshua Bengio、Aaron Courville
- 链接：<https://www.deeplearningbook.org/>
- 关键词：深度学习基础、神经网络、优化、生成模型
- 定位：业内俗称“花书”的经典深度学习教材

经典但部分内容相对老一些，适合建立深度学习的基础理论框架。

#### 5. Understanding Deep Learning

- 作者：Simon Prince
- 链接：<https://udlbook.github.io/udlbook/>
- 关键词：现代深度学习、Python notebook、可视化解释
- 定位：更新、更友好的深度学习教材

相比花书更现代，也更适合结合 notebook 实践学习。

---

### 三、强化学习

#### 6. Reinforcement Learning: An Introduction

- 作者：Richard Sutton、Andrew Barto
- 链接：<http://incompleteideas.net/book/the-book-2nd.html>
- 关键词：强化学习基础、价值函数、策略、TD 学习、Q-learning
- 定位：强化学习圣经

做 Agent、强化学习、RLHF / RLAIF 方向都值得读。

#### 7. Distributional Reinforcement Learning

- 链接：<https://www.distributional-rl.org/>
- 关键词：分布式强化学习、回报分布、Bellman 分布算子
- 定位：系统介绍 Distributional RL 的教材

适合在掌握基础 RL 后，进一步理解分布式视角下的强化学习。

#### 8. Multi-Agent Reinforcement Learning

- 链接：<https://www.marl-book.com/>
- 关键词：多智能体强化学习、协作、竞争、博弈
- 定位：多 Agent 协作与强化学习入门教材

适合学习多 Agent 系统、协作策略、博弈环境下的智能体训练。

---

### 四、应用与工程

#### 9. Foundations of Computer Vision

- 作者：Antonio Torralba 等
- 链接：<https://visionbook.mit.edu/>
- 关键词：计算机视觉、图像理解、现代 CV
- 定位：现代计算机视觉全景教材

适合系统学习 CV 基础和现代视觉模型脉络。

#### 10. Machine Learning Systems

- 链接：<https://mlsysbook.ai/>
- 关键词：机器学习系统、训练系统、推理系统、工程化
- 定位：ML 模型系统工程化教材

适合关注模型训练、推理、部署、性能优化、系统设计的人。

#### 11. Machine Learning in Production

- 链接：<https://mlip-cmu.github.io/book/>
- 关键词：MLOps、模型上线、生产系统、监控、数据漂移
- 定位：从模型到产品的工程实践教材

适合学习如何把模型真正部署到生产环境，而不是停留在 notebook 或实验阶段。

#### 12. Fairness and Machine Learning

- 链接：<https://fairmlbook.org/>
- 关键词：AI 公平性、偏见、伦理、算法影响
- 定位：AI 公平性问题教材

适合理解模型在现实社会系统中的偏见、公平性和风险问题。

---

## 推荐学习路线

### 路线 A：机器学习基础路线

1. Mathematics for Machine Learning
2. Foundations of Machine Learning
3. Probabilistic Machine Learning Part 2

适合目标：打牢 ML 理论基础。

### 路线 B：深度学习路线

1. Mathematics for Machine Learning
2. Understanding Deep Learning
3. Deep Learning

适合目标：系统理解神经网络和现代深度学习。

### 路线 C：Agent / 强化学习路线

1. Reinforcement Learning: An Introduction
2. Distributional Reinforcement Learning
3. Multi-Agent Reinforcement Learning

适合目标：学习 RL、Agent、多智能体协作。

### 路线 D：工程与落地路线

1. Machine Learning Systems
2. Machine Learning in Production
3. Foundations of Computer Vision / Fairness and Machine Learning

适合目标：从模型训练、推理、部署到产品化落地。

---

## 我的阅读优先级建议

如果结合 LLM / Agent / 训练系统方向，建议优先看：

1. **Mathematics for Machine Learning**：补齐数学底座
2. **Understanding Deep Learning**：建立现代深度学习框架
3. **Machine Learning Systems**：理解训练与推理系统
4. **Reinforcement Learning: An Introduction**：理解 Agent / RL 基础
5. **Machine Learning in Production**：理解模型生产化

如果已经有深度学习基础，可以直接从：

- Machine Learning Systems
- Reinforcement Learning: An Introduction
- Understanding Deep Learning

开始。

## 备注

原文提到“MIT 静悄悄放了 12 本 AI 教材出来，全部免费；同样内容去美国念个学位，学费要 5 万美元”。这里更适合理解为：这些公开教材构成了一套非常接近系统 AI 学位课程的自学资料组合，但学习效果仍取决于数学基础、编程实践和持续投入。