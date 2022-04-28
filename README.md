# AIOps工具包（aaarrgh）

## 异常检测算法论文

|名字|主要内容|时间|级别|特征|备注|
|-------|---------|---------|-------|---------|---------|
|FUNNEL: Assessing Software Changes in Web-based Services|主要方法名称交FUNNEL，其中主要通过iSST方法来讲变更前后的数据进行变点检测，改进了SST的传统方法，增加了计算效率提高了鲁棒性|2016|-|奇异光谱转化、SST|清华netman出品|
|Anomaly Detection in Streams with Extreme Value Theory|利用极值理论，从POT算法升级到SPOT和DSPOT算法，针对对概念迁移的处理，另外通过grimshaw方法对极值分布的二参数优化，并提出了一些策略提升优化效率|2017|KDD|EVT，grimshaw


## 根因定位论文
|名字|主要内容|时间|级别|特征|备注|
|-------|---------|---------|-------|---------|---------|
|MicroRCA: Root Cause Localization of Performance Issues in Microservices|利用调用链和因果关系发现，共同发现可能的微服务中的问题根因。可在线实时进行定位，并通过传播分析可关联服务和资源。|2020|-|birch，personality PageRank，故障传播图生成||
ß