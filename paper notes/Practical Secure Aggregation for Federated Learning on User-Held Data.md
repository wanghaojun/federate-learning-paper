**Practical Secure Aggregation for Federated Learning on User-Held Data  **

Keith Bonawitz*, Vladimir Ivanov*, Ben Kreuter*, Antonio Marcedoney*, H. Brendan McMahan*, Sarvar Patel*, Daniel Ramage*, Aaron Segal*, and Karn Seth*  

​                                                                                                                                                                                                **by Liu Yangchen**

[toc]

### 1 Introduction  

安全聚合是一类安全多方计算算法，其中一组互不信任的方$u \in \mathcal{U}$各自拥有一个私有值$x_u$并协作以计算一个聚合值，例如$\sum_{u \in \mathcal{U}} x_{u}$之和，而不会彼此泄露任何信息 关于他们的私人价值，但可以从总价值本身中学到的东西除外。 在这项工作中，我们考虑在联合学习模型中训练深层神经网络，在移动设备上的用户持有的训练数据上使用分布式梯度下降，并使用安全聚合来保护每个用户的模型梯度的私密性。 我们确定了效率和鲁棒性要求的结合，据我们所知，文献中现有的算法未能满足这些要求。 我们着手设计一种新颖的，通信效率高的安全聚合协议，用于处理高维数据，最多可容纳1 /3个未能完成该协议的用户。 对于16位输入值，我们的协议为$2^{10}$个用户和$2^{20}$维向量提供了1.73×通信扩展，为$2^{14}$个用户和$2^{24}$维向量提供了1.98×通信扩展。

### 2 Secure Aggregation for Federated Learning  

考虑培训一个深度神经网络，以预测用户在撰写短信时将键入的下一个单词，以提高手机屏幕键盘的键入准确性[11]。 建模者可能希望在大量用户的所有文本消息上训练这种模型。 但是，短信经常包含敏感信息。 用户可能不愿意将其副本上传到建模者的服务器。 取而代之的是，我们考虑在联邦学习设置中训练这样的模型，其中每个用户都在自己的移动设备上安全地维护她的短信的私有数据库，并在中央服务器的协调下基于高度处理的知识对共享的全局模型进行训练 ，范围最小的用户临时更新[14，17]。

神经网络表示将输入$x$映射到输出$y$的函数$f(x, \Theta)=y$，其中$f$由高维矢量$\Theta\in\mathbb{R}$参数化。 为了建模文本消息的组成，$x$可能会编码到目前为止输入的单词，$y$可能会编码下一个单词的概率分布。 一个训练示例是观察对$\langle x, y\rangle$和训练集是集合$D=\left\{\left\langle x_{i}, y_{i}\right\rangle ; i=1, \ldots, m\right\}$ . 我们定义训练集上的损失$\mathcal{L}_{f}(D, \Theta)=\frac{1}{|D|} \sum_{\left\langle x_{i}, y_{i}\right\rangle \in D} \mathcal{L}_{f}\left(x_{i}, y_{i}, \Theta\right)$，其中$\mathcal{L}_{f}(x,y, \Theta)=$$\ell(y, f(x, \Theta))$对于损失函数$\ell$，例如$\ell(y, \hat{y})=(y-\hat{y})^{2}$。 训练包括寻找参数$\Theta$来减小$\mathcal{L}_{f}(D, \Theta)$，通常使用变型小批量随机梯度下降法[4，10]。

在“联邦学习”设置中，每个用户$u \in \mathcal{U}$拥有一个私人的训练示例集$D_u$，其中$D=\bigcup_{u \in \mathcal{U}} D_{u}$。 为了进行随机梯度下降，对于每次更新，我们从一个随机子集$\mathcal{U}'\subset \mathcal{U}$中选择数据，并形成一个（虚拟）小批量$B=\bigcup_{u \in \mathcal{U}'} D_{u}$（实际上我们可以说$\left|\mathcal{U}^{\prime}\right|=10^{4}$而$\left|\mathcal{U}\right|=10^{7}$; 我们可能只考虑一个 每个用户本地数据集的子集）。 最小批量损失梯度$\nabla \mathcal{L}_{f}(B, \Theta)$可以重写为用户的加权平均值：$\nabla \mathcal{L}_{f}(B, \Theta)=\frac{1}{|B|} \sum_{u \in \mathcal{U}^{\prime}} \delta_{u}^{t}$其中$\delta_{u}^{t}=\left|D_{u}\right| \nabla \mathcal{L}_{f}\left(D_{u}, \Theta^{t}\right)$。 因此，用户只能共享$\left\langle\left|D_{u}\right|, \delta_{u}^{t}\right\rangle$； 与服务器一起，可以从中获取梯度下降步骤$\Theta^{t+1} \leftarrow \Theta^{t}-\eta \frac{\sum_{u \in \mathcal{U}^{\prime}} \delta_{u}^{t}}{\sum_{u \in \mathcal{U}^{\prime}}\left|D_{u}\right|}$。

虽然每次更新$\left\langle\left|D_{u}\right|, \delta_{u}^{t}\right\rangle$是短暂的，并且包含的信息少于原始$D_{u}$，用户可能仍想知道还剩下什么信息。 有证据表明，受过训练的神经网络的参数有时可以重建训练示例[8，17，1]; 参数更新是否可能遭受类似攻击？ 例如，如果输入$x$是编码最近键入的单词的单词词汇长度向量，则通用神经网络体系结构将在每个单词$w$的$Θ$中包含至少一个参数$θ_w$，以使得$\frac{\partial \mathcal{L}_{f}}{\partial \theta_{w}}$为非零 仅当$x$编码$w$时。因此，将通过检查$\delta_{u}^{t}$的非零条目来揭示$D_u$中最近键入的单词的集合。 但是，服务器不需要检查任何个人用户的更新； 它仅需要求和$\sum_{u \in \mathcal{U}}\left|D_{u}\right|$和$\sum_{u \in \mathcal{U}}\delta_{u}^{t}$。 使用安全聚合协议将确保服务器仅了解$U$中的一个或多个用户写了单词$w$，而不了解哪个用户。

联邦学习系统面临一些实际挑战。 移动设备只能零星地访问电源和网络连接，因此参与每个更新步骤的集合$\mathcal{U}$是不可预测的，并且系统必须对退出用户具有鲁棒性。 因为$Θ$可能包含数百万个参数，所以更新$\delta_{u}^{t}$可能很大，这对计量网络计划的用户而言是直接的成本。 移动设备通常还不能与其他移动设备建立直接通信通道（依靠服务器或服务提供商来调解此类通信），也不能本地认证其他移动设备。 因此，联邦学习激发了对安全聚合协议的需求，该协议：（1）在高维向量上运行；（2）即使在每个实例上都有一组新颖的用户，其通信效率也很高；（3）对用户丢弃具有鲁棒性 （4）在服务器介导的未经身份验证的网络模型的约束下提供了最强的安全性。

### 3 A Practical Secure Aggregation Protocol  

在我们的协议中，有两种类型的参与方：单个服务器$S$和$n$个用户$\mathcal{U}$的集合。每个用户$u\in\mathcal{U}$都拥有维度$k$的私有向量$x_u$。 我们假设$x_u$和$\sum_{u \in \mathcal{U}}x_{u}$的所有元素都是[0;  $R$）为一些已知的$R$。 正确性要求，如果各方都是诚实的，则$S$对用户$\bar{u} \subseteq \mathcal{U}$的其中$\left|\overline{\mathcal{U}}\right|≥\frac{n}{2}$的用户子集学习$\bar{x}=\sum_{u \in \overline{\mathcal{U}}} x_{u}$。 安全性要求（1）$S$除了从$\bar{x}$可以推断出的东西外，什么都学不到；（2）每个用户$u\in\mathcal{U}$什么都学不到。
   我们考虑了三种不同的威胁模型。 在所有这些人中，所有用户都诚实地遵守该协议，但是服务器可能尝试以不同的方式学习额外的信息：

**（T1）**服务器是诚实但好奇的，也就是说，它诚实地遵循协议，但是尝试从从用户那里收到的消息中学习尽可能多的东西。
**（T2）**服务器可以欺骗其他用户已退出的用户，包括在不同用户之间不一致地报告退出。
**（T3）**服务器可以撒谎是谁退出（就像在T2中一样），并且还可以访问某些数量有限的用户（他们自己诚实地遵循协议）的私有内存。  （在这种情况下，隐私权要求仅适用于其余用户的输入。）

**Protocol 0: Masking with One-Time Pads**   我们通过一系列改进来开发协议。 我们首先假设所有各方都已完成协议并拥有具有足够带宽的成对安全通信通道。 每对用户首先同意匹配的一对输入扰动。即，用户$u$从$[0, R)^{k}$均匀地采样向量$s_{u,v}$为每个其他用户$v$。用户$u$和$v$在其安全通道上交换$s_{u,v}$和$s_{v,u}$并计算扰动$p_{u,v} =s_{u,v} -s_{v,u}（mod R）$，请注意$p_{u,v}= -p_{v,u}（mod R）$并在$u = v$时取$p_{u,v}=0$。每个用户发送到服务器：$y_u=x_u+\sum_{v \in \mathcal{U}}p_{u,v}（mod R）$。 服务器仅将干扰值相加：$\bar{x}=\sum_{u \in \mathcal{U}}y_{u}（mod R）$。 正确性得到保证，因为$y_u$中的配对扰动会抵消：
$$
\bar{x}=\sum_{u \in \mathcal{U}} x_{u}+\sum_{u \in \mathcal{U}} \sum_{v \in \mathcal{U}} p_{u, v}=\sum_{u \in \mathcal{U}} x_{u}+\sum_{u \in \mathcal{U}} \sum_{v \in \mathcal{U}} s_{u, v}-\sum_{u \in \mathcal{U}} \sum_{v \in \mathcal{U}} s_{v, u}=\sum_{u \in \mathcal{U}} x_{u} \quad(\bmod R)
$$
协议0为用户保证了完美的隐私； 由于对用户添加的$s_{u, v}$因子进行了统一采样，因此$y_u$值在服务器上均匀地随机出现，受到$\bar{x}=\sum_{u \in \mathcal{U}}y_{u}（mod R）$的约束。 实际上，即使服务器可以访问某些用户的内存，其余用户的隐私也会保留。

**Protocol 1: Dropped User Recovery using Secret Sharing**    不幸的是，协议0未能通过我们的几项设计标准，包括鲁棒性：如果任何用户$u$无法通过将其$y_u$发送到服务器来完成协议，则结果总数将被$y_u$会取消的扰动所掩盖。为了实现鲁棒性，我们首先向协议中添加了一个初始回合，在该协议中，用户$u$生成了公共/专用密钥对，并在成对通道上广播了公用密钥。 从$u$到$v$的所有未来消息将由服务器进行中间处理，但使用$v$的公钥加密，并由$u$签名，以模拟经过安全验证的渠道。 这使服务器可以保持一致的观点，即哪些用户成功通过了协议的每一轮。  （在这里，我们暂时假设服务器忠实地在用户之间传递所有消息。）

在选择$s_{u,v}$值之后，我们还会在用户之间添加秘密共享回合。 在这一轮中，每个用户使用$（t; n）$阈值方案（例如Shamir的Secret Sharing [16]）对$t> \frac{n}{2}$的每个扰动$p_{u,v}$计算$n$个份额。 对于每个$u$用户，她使用每个用户$v$的公钥加密一个共享，然后将所有这些共享传递给服务器。 服务器从大小至少为$t$的用户$\mathcal{U}_1⊆\mathcal{U}$的子集收集份额（例如，通过等待一段固定的时间），然后考虑所有其他用户掉线。 服务器向每个用户$v∈\mathcal{U}_1$传递为该用户加密的秘密共享。 现在，$\mathcal{U}_1$中的所有用户都从接收到的共享集推断出尚存的用户集$\mathcal{U}_1$的一致视图。 当用户计算$y_u$时，它仅包括与尚存的用户相关的那些扰动。 即$y_u=x_u+\sum_{v \in \mathcal{U}_1}p_{u,v}（mod R）$。

服务器从至少$t$个用户$\mathcal{U}_2⊆\mathcal{U}_1$接收到$y_u$后，考虑到所有其他用户都将被丢弃，它将继续进行新的非屏蔽回合。 服务器从$\mathcal{U}_2$中的其余用户请求$\mathcal{U}_1 \backslash\mathcal{U}_2$中已删除用户生成的所有秘密份额。 只要$\left|\mathcal{U}_2\right|>t$，每个用户都将使用这些份额进行响应。 一旦服务器从至少$t$个用户处接收到份额，它便会重建$\mathcal{U}_1 \backslash\mathcal{U}_2$的扰动并计算总值：$\bar{x}=\sum_{u \in \mathcal{U}_{2}} y_{u}-\sum_{u \in \mathcal{U}_{2}} \sum_{v \in \mathcal{U}_{1} \backslash \mathcal{U}_{2}} p_{u, v} \quad(\bmod R)$。 只要至少$t$个用户完成协议，就可以保证$\bar{\mathcal{U}}=\mathcal{U}_2$的正确性。 在这种情况下，和$\bar{x}$包括至少$t> \frac{n}{2}$个用户的值，并且所有扰动都被抵消：
$$
\bar{x}=\left(\sum_{u \in \mathcal{U}_{2}} x_{u}+\sum_{u \in \mathcal{U}_{2}} \sum_{v \in \mathcal{U}_{1}} p_{u, v}\right)-\sum_{u \in \mathcal{U}_{2}} \sum_{v \in \mathcal{U}_{1} \backslash \mathcal{U}_{2}} p_{u, v}=\sum_{u \in \mathcal{U}_{2}} x_{u}+\sum_{u \in \mathcal{U}_{2}} \sum_{v \in \mathcal{U}_{2}} p_{u, v}=\sum_{u \in \mathcal{U}_{2}} x_{u} \quad(\bmod R)
$$
但是，安全性已丢失：如果服务器由于疏忽（例如，$y_u$到来太晚）或出于恶意而从$\mathcal{U}_2$错误地忽略了$u$，则$\mathcal{U}_2$中的诚实用户将为服务器提供删除所有服务器所需的所有秘密共享。 遮蔽了$y_n$中$x_n$的微扰。 这意味着即使对于诚实但好奇的服务器（威胁模型T1），我们也无法保证安全。

**Protocol 2: Double-Masking to Thwart a Malicious Server  ** 为了确保安全，我们引入了双重屏蔽结构，即使服务器可以重建$u$的扰动，它也可以保护$x_u$。首先，每个用户$u$从$[0, R)^{k}$均匀采样附加随机值$b_u$与$s_{u, v}$值的生成在同一轮中。 在秘密共享回合中，用户还生成$b_u$的份额并将其分配给其他每个用户。 生成$y_u$时，用户还添加以下辅助掩码：$y_{u}=x_{u}+b_{u}+\sum_{v \in \mathcal{U}_{1}} p_{u, v}(\bmod R)$。 在取消屏蔽回合期间，服务器必须针对每个用户$u∈\mathcal{U}_1$做出明确选择：从每个尚存成员$v∈\mathcal{U}_2$中，服务器可以请求$p_{u,v}$的份额扰动或与$u$相关的份额$ b_u$ ; 诚实用户$v$仅在$\left|\mathcal{U}_2\right|>t$时才响应，并且永远不会为同一用户显示这两种份额。 在收集了所有$u∈\mathcal{U}_1 \backslash\mathcal{U}_2$的至少$t$份$p_{u,v}$和所有$u∈\mathcal{U}_2$的$b_u$的$t$份后，服务器将重构秘密并计算出总值：$\bar{x}=\sum_{u \in \mathcal{U}_{2}} y_{u}-\sum_{u \in \mathcal{U}_{2}} b_{u}-\sum_{u \in \mathcal{U}_{2}} \sum_{v \in \mathcal{U}_{1} \backslash \mathcal{U}_{2}} p_{u, v}(\bmod R)$。

现在，我们可以在威胁模型T1中保证$t> \frac{n}{2}$的安全性，因为$x_u$始终被$p_{u, v}$s或$b_u$s掩盖。 可以看出，在威胁模型T2和T3中，阈值必须相应地提高到$\frac{2n}{3}$和$\frac{4n}{5}$。 我们将详细分析以及任意恶意和共谋的服务器和用户的情况推迟到完整版本。

**Protocol 3: Exchanging Secrets Efficiently  ** 协议2在选择正确的$t$的情况下既稳健又安全，但它需要$O（kn^2）$通信，在协议的这种改进中我们要解决这个问题。

观察到一个秘密值可以通过使用它来播种密码安全伪随机发生器（PRG）来扩展为伪随机值向量[2，9]。 因此，我们可以只生成标量种子$s_{u,v}$和$b_u$，并将它们扩展为$k$个元素向量。 尽管如此，每个用户与其他用户都有（n − 1）个秘密$s_{u,v}$，并且必须发布所有这些秘密的份额。 我们使用密钥协议来更有效地建立这些秘密。 每个用户都会生成一个Diffie-Hellman私钥$s^{S K}$和公钥$s^{P K}$。用户将其公钥发送到服务器（已根据协议1进行了身份验证）。 然后，服务器将所有公钥广播给所有用户，并为其保留一个副本。 每对用户$u,  v$现在可以同意一个秘密$s_{u, v}=s_{v, u}=\operatorname{AGREE}\left(s_{u}^{S K}, s_{v}^{P K}\right)=\operatorname{AGREE}\left(s_{v}^{S K}, s_{u}^{P K}\right)$。 为了构造扰动，我们假设$\mathcal{U}$的总阶，对于$u <v$，$p_{u, v} = PRG（s_{u, v}）$；对于$u> v$，$p_{u, v} = -PRG（s_{u, v}）$，并且$p_{u,v} = 0 $对于$u = v$（与以前一样）。 现在，服务器只需学习$s_{u}^{S K}$即可重建所有$u$扰动； 因此，在秘密共享回合中，您只需要分发$s_{u}^{S K}$和$b_u$的份额。 在每种不同的威胁模型中，可以证明协议3的安全性与协议2的安全性基本相同。

**Protocol 4: Minimizing Trust in Practice  ** 协议3实际上缺乏针对移动设备的部署，因为它们缺少成对的安全通信和身份验证。 我们建议通过用服务器介导的密钥协议替换协议1中描述的公钥/私钥的交换来引导通信协议，其中每个用户生成一个Diffie-Hellman私钥$c^{SK}$和公钥$c^{PK}$，并将它们与一起发布 $s^{P K}$ .我们立即注意到，服务器现在可能会进行中间人攻击，但认为这是可以容忍的，原因有几个。 首先，对于缺少身份验证机制或预先存在的公钥基础结构的用户而言，这是不可避免的。 仅依靠引导程序回合的非恶意性也构成对信任的最小化：实现此阶段的代码很小，可以公开审核，外包给受信任的第三方，或者通过提供远程证明功能的受信任的计算平台来实现[  7、6、18]。 此外，该协议可显着提高安全性（通过防御服务器主动恶意攻击以外的任何措施）并提供前向机密性（密钥交换后的任何时间损害服务器，即使所有数据和通信也不会给攻击者带来任何好处） 已被完全记录）。

我们在表1中总结了该协议的性能。假定密钥协议的公共密钥和加密的秘密共享为256位，并且用户的输入都在相同的范围内 $[0,  R_U-1]$，与发送原始向量相比，每个用户传输$\frac{256(7 n-4)+k\left\lceil\log _{2}\left(n\left(R_{U}-1\right)+1\right)\right\rceil+n}{k\left\lceil\log _{2} R_{U}\right\rceil}$的数据更多。

### 4 Related work  

安全聚合的受限情况（其中除一个用户外，所有用户都具有输入0）可以表示为就餐密码学家网络（DC-net），它通过使用输入的成对盲法提供匿名性[3，9]，从而可以无可奈何地学习每个 用户的输入。 最近的研究已经检查了恶意用户在场时的通信效率和操作[5]。 但是，即使一个用户中止得太早，现有协议也必须从头开始，这可能会非常昂贵[13]。已经探索了基于模加法的加密方案中的成对盲，但是现有方案既不能有效用于矢量，也不能可靠地解决单个故障[2，12]。 其他方案（例如    基于Paillier密码系统[15]的计算非常昂贵。

![](http://raw.githubusercontent.com/beichen777/paperimage/main/PracticalSecureAggregationforFederatedlearningonUser-HeldDatatable1.PNG)

--------------------

**表1：**协议4成本摘要（派生至全文）。

![](https://raw.githubusercontent.com/beichen777/paperimage/main/PracticalSecureAggregationforFederatedlearningonUser-HeldDatafigure1.PNG)

----------------

**图1：**协议4通讯图



