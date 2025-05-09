from flask import Flask, request
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API key
genai.configure(api_key="AIzaSyBMyqVdwhbJ3OjLpNVfXs_8oAQ4YEOZhSw")
model = genai.GenerativeModel('gemini-2.0-flash')

# Route 1: Accepts a POST request with a question and returns Gemini's response
@app.route('/query', methods=['POST'])
def ask():
    question = request.data.decode("utf-8")
    response = model.generate_content(question)
    return response.text

# Route 2: Returns a static paragraph of simple text
@app.route('/intro', methods=['GET'])
def intro():
    return """ import java.util.*;

public class Graph{
     
    public Map<Integer , List<Integer>> g = new HashMap<>();

    public void add(int a,int b){
        g.putIfAbsent(a , new ArrayList<>());

        if(a!=b){
            g.putIfAbsent(b, new ArrayList<>());
            g.get(a).add(b);
            g.get(b).add(a);
        }
    }

    public void dfs(int start , Set<Integer> visited){
          visited.add(start);
          System.out.print(start + " ");

          for(int member : g.get(start) ){
              if(!visited.contains(member)){
                  dfs(member , visited);
              }
          }
    }


    public void bfs(int start){
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> q = new LinkedList<>();
        
        visited.add(start);
        q.add(start);
        while(!q.isEmpty()){
            int v = q.poll();
            System.out.print(v+" ");
            for(int member : g.get(v)){
                if(!visited.contains(member)){
                     visited.add(member);
                     q.add(member);
                }
            }
        }
    }

    public static void main(String[] args){
        Graph g = new Graph();

        g.add(1,2);
        g.add(1,3);
        g.add(1,4);
        g.add(3,6);
        g.add(3,5);
        g.add(6,7);

        System.out.println("Depth first Search :");
        g.dfs(1 , new HashSet<>());
        System.out.println("Breadth first Search :");
        g.bfs(1);

        
    }
}














import java.util.*;

class Job implements Comparable<Job>{
    
    char id;
    int deadline;
    int profit;

    public Job(char i,int d , int p){
        id  =i;
        deadline = d;
        profit = p;
    }

    @Override 
    public int compareTo(Job other){
        return other.profit - this.profit;
    }
}

public class JobSchedule {
    public static void main(String[] args) {
        
        Job[] jobs = {
            new Job('a',2,100),
            new Job('b',1,19),
            new Job('c',2,27),
            new Job('d',1, 25),
            new Job('e', 3,15)
        };

        Arrays.sort(jobs);
        int n = jobs.length;
        boolean[] slot = new boolean[n];
        char[] result = new char[n];
        
        
        for (int i = 0; i < result.length; i++) {
            for (int j = Math.min(n-1, jobs[i].deadline-1); j >=0; j--) {
                if(!slot[j]){
                    slot[j] = true;
                    result[j] = jobs[i].id;
                    break;
                }
            }
        }

        System.out.print("Scheduled Jobs: ");
        for (char job : result) {
            if (job != 0)
                System.out.print(job + " ");
        }
    }
}











import java.util.*;

class Edge implements Comparable<Edge> {
    int src,dest,weight;

    public Edge(int s,int d , int w){
        src = s;
        dest = d;
        weight = w;
    }
    
    public int compareTo(Edge other){
        return this.weight - other.weight;
    }
}

class Kruskal {
    int[] parent;
    
    int find(int i){
        if(i != parent[i]){
            parent[i] = find(parent[i]);
        }

        return parent[i];
    }

    void union(int u , int v){
        parent[find(u)] = find(v);
    }


    void kruskalMST(List<Edge> edges , int V){
        Collections.sort(edges);
        parent = new int[V];

        for (int i =0 ; i<V;i++) parent[i] = i;

        System.out.println("Edges : weight");

        for (Edge e : edges){
            int uset = find(e.src);
            int vset = find(e.dest);

            if(uset != vset){
                System.out.println(e.src+"-"+e.dest+":"+e.weight);
                union(uset, vset);
            }
        }
    }


    public static void main(String[] args) {
        List<Edge> edges = Arrays.asList(
            new Edge(0, 1, 10),
            new Edge(0, 2, 6),
            new Edge(0, 3, 5),
            new Edge(1, 3, 15),
            new Edge(3, 3, 4)
        );

        new Kruskal().kruskalMST(edges, 4);
    }
}












import java.util.*;

public class Prims {
    public static void main(String[] args) {
        
        int V = 5;
        int[][] graph = {
            {0,2,0,6,0},
            {2,0,3,8,5},
            {0,3,0,0,7},
            {6,8,0,0,9},
            {0,5,7,9,0}
        };

        boolean[] selected = new boolean[V];
        selected[0] = true;
        int edges = 0;

        while(edges < V-1){
            int min = Integer.MAX_VALUE;
            int x = -1;
            int y = -1;

            for(int i = 0 ; i < V ;i++){
                if(selected[i]){
                    for(int j =0; j< V ;j++){
                        if(!selected[j] && graph[i][j] != 0 && graph[i][j] < min ){
                            min = graph[i][j];
                            x = i;
                            y = j;
                        }
                    }
                }
            }
            
            System.out.println(x+"-"+y+":"+graph[x][y]);
            selected[y] = true;
            edges++;

        }
    }
}









public class NQueens {
    static int N = 8;

    static boolean isSafe(int board[][], int row, int col) {
        for (int i = 0; i < col; i++)
            if (board[row][i] == 1) return false;

        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
            if (board[i][j] == 1) return false;

        for (int i = row, j = col; i < N && j >= 0; i++, j--)
            if (board[i][j] == 1) return false;

        return true;
    }

    static boolean solveNQUtil(int board[][], int col) {
        if (col >= N) return true;

        for (int i = 0; i < N; i++) {
            if (isSafe(board, i, col)) {
                board[i][col] = 1;
                if (solveNQUtil(board, col + 1)) return true;
                board[i][col] = 0;
            }
        }
        return false;
    }

    static void solveNQ() {
        int board[][] = new int[N][N];
        if (!solveNQUtil(board, 0)) {
            System.out.println("No Solution");
            return;
        }

        for (int[] row : board) {
            for (int val : row) {
                System.out.print((val == 1 ? "Q " : ". "));
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        solveNQ();
    }
}





import java.util.*;

class Node implements Comparable<Node> {
    int x, y, cost, heuristic, f;
    Node parent;

    Node(int x, int y, int cost, int heuristic, Node parent) {
        this.x = x;
        this.y = y;
        this.cost = cost;
        this.heuristic = heuristic;
        this.f = cost + heuristic;
        this.parent = parent;
    }

    public int compareTo(Node o) {
        return this.f - o.f;
    }
}

public class Astar {
    static int[][] grid = {
        {0, 0, 0, 0},
        {1, 1, 0, 1},
        {0, 0, 0, 0},
        {0, 1, 1, 0}
    };
    static int rows = 4, cols = 4;
    static int[] dx = {0, 1, 0, -1};
    static int[] dy = {1, 0, -1, 0};

    static boolean isValid(int x, int y) {
        return x >= 0 && y >= 0 && x < rows && y < cols && grid[x][y] == 0;
    }

    static int heuristic(int x, int y, int gx, int gy) {
        return Math.abs(x - gx) + Math.abs(y - gy);
    }

    public static void main(String[] args) {
        int sx = 0, sy = 0, gx = 3, gy = 3;
        PriorityQueue<Node> open = new PriorityQueue<>();
        boolean[][] visited = new boolean[rows][cols];
        open.add(new Node(sx, sy, 0, heuristic(sx, sy, gx, gy), null));

        while (!open.isEmpty()) {
            Node current = open.poll();
            if (current.x == gx && current.y == gy) {
                while (current != null) {
                    System.out.println("(" + current.x + "," + current.y + ")");
                    current = current.parent;
                }
                break;
            }
            visited[current.x][current.y] = true;
            for (int i = 0; i < 4; i++) {
                int nx = current.x + dx[i], ny = current.y + dy[i];
                if (isValid(nx, ny) && !visited[nx][ny]) {
                    open.add(new Node(nx, ny, current.cost + 1, heuristic(nx, ny, gx, gy), current));
                }
            }
        }
    }
}











import java.util.*;

public class Dijkstra {
    static int V = 5;

    static void dijkstra(int[][] graph, int src) {
        int[] dist = new int[V];
        boolean[] visited = new boolean[V];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;

        for (int count = 0; count < V - 1; count++) {
            int u = minDistance(dist, visited);
            visited[u] = true;
            for (int v = 0; v < V; v++) {
                if (!visited[v] && graph[u][v] != 0 && dist[u] != Integer.MAX_VALUE &&
                    dist[u] + graph[u][v] < dist[v]) {
                    dist[v] = dist[u] + graph[u][v];
                }
            }
        }

        for (int i = 0; i < V; i++) {
            System.out.println("Distance from " + src + " to " + i + " is " + dist[i]);
        }
    }

    static int minDistance(int[] dist, boolean[] visited) {
        int min = Integer.MAX_VALUE, min_index = -1;
        for (int v = 0; v < V; v++) {
            if (!visited[v] && dist[v] <= min) {
                min = dist[v];
                min_index = v;
            }
        }
        return min_index;
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 10, 0, 0, 5},
            {0, 0, 1, 0, 2},
            {0, 0, 0, 4, 0},
            {7, 0, 6, 0, 0},
            {0, 3, 9, 2, 0}
        };
        dijkstra(graph, 0);
    }
}
"""
