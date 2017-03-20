import java.util.ArrayList;

public class Snake{
    public enum Direction{UP, DOWN, LEFT, RIGHT}
    private class SnakeBody{
        public int[] coor; //the x y coordinates of this piece

        public SnakeBody(int x, int y){
            this.x = x;
            this.y = y;
        }

        //Constructor that takes in the next piece of snake
        //to create this one flowing same direction one pos offset
        public SnakeBody(SnakeBody parent){
            this.x = parent.x-1;
            this.y = parent.y-1;
        }

        public int[] getCoor(){
            int[] coor = new int [2];
            coor[0] = x;
            coor[1] = y;
            return coor;
        }
    }

    private ArrayList<SnakeBody> head;
    private Direction currDir; //current direction the snake is moving
    private int gridSize; //size of the square grid
    private boolean lost; //flag to see if snake lost

    public Snake(int gridSize){
        this.gridSize = gridSize;
        head = new ArrayList<>();
        currDir = RIGHT;
        lost = false; 

        //initialize the snake
        int startPos = gridSize/2;
        for(int i = 0; i < 3; i++)
            head.add(new SnakeBody(startPos, startPos-i));
    }
    
    //increase size of snake by 1
    public void grow(){
        head.add(new SnakeBody(head.get(head.size()-1));           
    }

    public void move(){
        SnakeBody headBod = head.get(0);
        SnakeBody curr;
        SnakeBody next;

        int nextX;
        int nextY;
        
        //find the next x position
        if(currDir == UP || currDir == DOWN){
            nextX = currDir == UP ? headBod.x-1 : headBod.x+1;
            nextY = headBod.y;
        }else{
            nextX = headBod.x;
            nextY = currDir == LEFT ? headBod.y-1 : headBod.y+1;
        }
        
        //check to see if hit the wall
        if(nextX >= gridSize || nextX < 0 || nextY >= gridSize || nextY < 0)
            lost = true;
        
        //move the snake body
        for(int i = head.size()-1; i > 0; i--){
            curr = head.get(i);
            next = head.get(i-1);
            
            //move the snake
            curr.x = next.x;
            curr.y = next.y;

            //check to see if lost
            if(nextX == curr.x && nextY == curr.y)
                lost = true;
        }
        
        //set the new head position
        headBod.x = nextX;
        headBod.y = nextY;
    }
    
    public boolean getSize(){
        return head.size();
    }
    
    //gives back the coordinates of a piece
    public int[] getBodyCoord(int pos){
        if(pos < 0 || pos >= head.size())
            return null;
        
        return head.get(pos).getCoor();
    }
    
    public void changeDir(Direction dir){
        currDir = dir;
    }

    //tells if the snake lost or not
    public boolean lost(){
        return lost;
    }
}
