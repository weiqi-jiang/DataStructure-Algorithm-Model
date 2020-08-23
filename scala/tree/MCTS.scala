/*
"游戏规则"
双方每次从(1,2,-1,-2)中选择一个数，经过若干轮次，最终目标是数值求和尽可能接近1

*/

object MCTS {
  import scala.collection.mutable.ArrayBuffer
  import util.Random
  import math._

  // 最大游戏轮次
  val MAX_ROUND = 20
  // 选择项
  val AVAILABLE_CHOICES = Array(1,-1,2,-2)
  // 每次做决策时，做MCTS模拟的次数
  val NUM_SIMULATION = 100

  class State{
    /*
    当前游戏全局状态

    Attribute
    ------------
    game_value: 当前累计value值
    round： 当前游戏轮次
    choice_path: 记录到目前为止选择路径

    Method
    -------------
    getGameValue: 返回累计value值
    setGameValue： 指定game_value 值
    getRound：返回当前轮次
    setRound： 设置当前轮次
    isEnd： 游戏是否结束
    getChoicePath： 返回选择路径
    setChoicePath： 设置选择路径
    getReward： 返回奖励值，沿着选择路径反向传播
    getRandomNewState： 随机选择一个选项，进入游戏进入下一个状态，用于模拟过程
     */
    private var game_value = 0.0
    private var round = 0
    private var choice_path = Array[Int]()

    def getGameValue=game_value
    def setGameValue(value: Double) = {game_value = value}
    def getRound = round
    def setRound(r:Int)={round = r}
    def isEnd = round == MAX_ROUND
    def getChoicePath = choice_path
    def setChoicePath(choices: Array[Int]) = {choice_path = choices}
    def getReward = -abs(1-game_value)

    def getRandomNewState={
      val random_choice = AVAILABLE_CHOICES(Random.nextInt(AVAILABLE_CHOICES.size))
      val next_state = new State
      next_state.setGameValue(game_value + random_choice)
      next_state.setRound(round+1)
      next_state.setChoicePath(choice_path ++ Array(random_choice))
      next_state
    }

  }

  class Node{
    /*
    蒙特卡洛搜索树节点类

    Attribute
    ------------
    parent: 父节点
    children： 子节点
    visit_times： 访问次数
    quality_value： 节点当前quality 值
    state： 到当前节点未知，整局游戏的状态

    Method
    -------------
    setState: 设置状态
    getState： 返回状态
    setParent： 设置父节点
    getParent： 返回父节点
    getChildren： 返回子节点
    setVisitTimes： 设置访问次数
    getVisitTimes： 返回访问次数
    setQualityValue： 设置quality值
    getQualityValue： 返回quality值
    addChildren： 添加子节点
    isAllExpand： 当前节点是否完全扩展，也就是是否有四个不同选择的子节点
    */

    private var parent: Node = null
    private val children = ArrayBuffer[Node]()
    private var visit_times: Double = _
    private var quality_value: Double = _
    private var state: State = _

    def setState(s:State)={state = s}
    def getState = state
    def setParent(p:Node) = {parent = p}
    def getParent = parent
    def getChildren = children
    def setVisitTimes(v:Double)={visit_times = v }
    def getVisitTimes = visit_times
    def setQualityValue(q:Double) = {quality_value = q}
    def getQualityValue = quality_value
    def addChildren(child: Node) = {
      children.append(child)
      child.setParent(this)
    }
    def isAllExpand= children.length == AVAILABLE_CHOICES.length
  }


  def best_child(node: Node, is_exploration: Boolean)={
    /*
    ucb 算法，权衡探索和利用选择得分最高的子节点，如果是预测阶段直接返回当前平均收益最高的节点

    输入
    ---------------
    node:
    is_exploration: 是否是训练阶段

    输出
    ---------------
    best_child: 最佳子节点

     */
    var max_ucb = Double.NegativeInfinity
    var best_child: Node = new Node()

    val c = if(is_exploration) 1/sqrt(2.0)  else 0.0
    for(child<-node.getChildren){
      val part1 = child.getQualityValue/child.getVisitTimes
      val part2 = 2*log(node.getVisitTimes)/child.getVisitTimes
      val ucb = part1 + c*sqrt(part2)
      if(ucb > max_ucb){
        max_ucb = ucb
        best_child = child
      }
    }
    best_child
  }

  def expand(node:Node) = {
    /*
    在该节点上扩展出新的节点，随机选择action，需保证和当前节点下的所有子节点的action不同

    输入
    --------------
    node：当前节点

    输出
    --------------
    new_node: 新扩展出的节点
     */
    val existed_state = for(child<-node.getChildren) yield child.getState
    var new_state = node.getState.getRandomNewState
    while( existed_state.contains(new_state) ){new_state = node.getState.getRandomNewState}
    val new_node = new Node()
    new_node.setState(new_state)
    node.addChildren(new_node)
    new_node
  }

  def tree_policy(node:Node):Node={
    /*
    selection和expansion阶段，传入搜索起点，如果是叶子节点直接返回，如果没有被完全探索则随机选择，否则根据UCB值计算最佳节点
    并且递归进行选择过程直到未完全扩展或者游戏结束

    输入
    --------------
    node: 搜索起始节点

    输出
    --------------
    cur_node/new_node ：如果输入是叶子节点，则输出和输入一样；否则输出新扩展出来的节点
     */
    var cur_node = node
    while(cur_node.getState.isEnd == false){

      if(cur_node.isAllExpand){
        cur_node = best_child(node, true)
      }
      else {
        val new_node = expand(node)
        return new_node
      }
    }
    cur_node
  }

  def simulation(node: Node)={
    /*
    模拟游戏阶段，模拟随机选择，直到游戏结束，返回该次模拟的奖励值

    输入
    ------------------
    node: 模拟开始节点

    输出
    ------------------
    reward： 该次模拟的奖励值

     */
    var cur_state = node.getState
    while(cur_state.isEnd==false){
      cur_state = cur_state.getRandomNewState
    }
    cur_state.getReward
  }

  def backpropagation(node: Node, reward: Double)={
    /*
    反向传播, 直到根节点，路径上的节点访问次数+1，quality值+reward

    输入
    --------------
    node： 当前节点
    reward：模拟过程或者的奖励值

     */
    var cur_node = node
    while(cur_node != null){
      cur_node.setVisitTimes(node.getVisitTimes+1)
      cur_node.setQualityValue(node.getQualityValue + reward)
      cur_node = cur_node.getParent
    }
  }

  def MCTS(node: Node)={
    /*
    MCTS 搜素过程, 从根节点出发，每次完整的经历整个搜素过程，获得最佳下一步
    搜素过程：
        1. selection 和 expansion找到扩展节点
        2. 在扩展节点上模拟规定次数，并反向传播，更新节点的quality值
        3. 反向传播后

    输入
    -----------------
    node： 当前节点

    输出
    -----------------
    best_child: 基于当前节点更新后的quality值，UCB计算出的最佳下一步
     */
    for(i <- 1 to NUM_SIMULATION){
      val expand_node = tree_policy(node)
      val reward = simulation(expand_node)
      backpropagation(expand_node, reward)
    }
    best_child(node,false)
  }


  def main(args: Array[String]): Unit = {
    /*
    主入口函数
     */
    val init_state = new State
    val init_node = new Node
    var cur_node = init_node
    init_node.setState(init_state)


    for( i <-1 to MAX_ROUND){
      println(s"Play Round: $i")
      cur_node = MCTS(cur_node)

      println(cur_node.getState.getChoicePath.mkString(sep=" "))
      println(cur_node.getState.getChoicePath.sum)
    }

  }
}
