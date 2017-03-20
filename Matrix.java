public class Matrix{
    private int[][] mtrx;
    
    public static Matrix mult(Matrix mx1, Matrix mx2){
        //check for valid input
        if(mx1 == null || mx2 == null)
            return null;

        //get the int datafields to make multiplication easier
        int[][] m1 = mx1.getMtrx();
        int[][] m2 = mx2.getMtrx();

        //check if can mulitply together
        if(m1[0].length != m2.length)
            return null;
        
        return new Matrix(multInt(m1, m2));
    }

    //used to multiply 2 int arrays togther using matrix mult
    private static int[][] multInt(int[][] mx1, int[][] mx2){
        //allocate mem for the result
        int[][] res = new int[mx1.length][mx2[0].length];

        for(int i = 0; i < res.length; i++){
            for(int j = 0; j < res[i].length; j++){
                //get value for spot i j
                int val = 0;
                for(int z = 0; z < m2.length; z++)
                    val += mx1[i][z]*mx2[z][j];

                //put value in
                res[i][j] = val;
            }
        }

        return res;
    }

    public Matrix(int col, int row){
        mtrx = new int [col][row];
    }

    public Matrix(int[][] intMtrx){
        mtrx = intMtrx;
    }

    public int[][] getMtrx(){
        return mtrx;
    }

    public boolean mult(Matrix other){
        //check for good flags
        if(other == null)
            return false;

        int[][] otherInt = other.getMtrx();

        if(mtrx[0].length != otherInt.length)
            return false;

        mtrx = multInt(mtrx, otherInt);
        return true;
    }
}
