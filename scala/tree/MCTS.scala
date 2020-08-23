
object MCTS {
  import scala.collection.mutable.ArrayBuffer
  import util.Random
  import math._

  val MAX_ROUND = 10
  val AVAILABLE_CHOICES = Array(1,-1,2,-2)
  val NUM_SIMULATION = 20

  class State{
    /*
    记录当前游戏的状态
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
    ucb 算法，权衡探索和利用选择得分最高的节点，如果是预测阶段直接返回当前平均收益最高的节点
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
    input： 当前节点
    func： 在该节点上扩展出新的节点，随机选择action，需保证和当前节点下的所有子节点的action不同
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
    var cur_state = node.getState
    while(cur_state.isEnd==false){
      cur_state = cur_state.getRandomNewState
    }
    cur_state.getReward
  }

  def backpropagation(node: Node, reward: Double)={
    var cur_node = node
    while(cur_node != null){
      cur_node.setVisitTimes(node.getVisitTimes+1)
      cur_node.setQualityValue(node.getQualityValue + reward)
      cur_node = cur_node.getParent
    }
  }

  def MCTS(node: Node)={
    for(i <- 1 to NUM_SIMULATION){
      val expand_node = tree_policy(node)
      val reward = simulation(expand_node)
      backpropagation(expand_node, reward)
    }
    best_child(node,false)
  }


  def main(args: Array[String]): Unit = {
    val init_state = new State
    val init_node = new Node
    init_node.setState(init_state)
    var cur_node = init_node

    for( i <-1 to MAX_ROUND){
      println(s"Play Round: $i")
      cur_node = MCTS(cur_node)

      println(cur_node.getState.getChoicePath.mkString)
    }

  }
}
