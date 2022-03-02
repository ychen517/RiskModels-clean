import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * @author rkannappan
 */
public class RunSQLScript {
   
   protected final static String DRIVER = "org.apache.derby.jdbc.EmbeddedDriver";
   
    public RunSQLScript(String scriptName) {
	try {
	    InputStream file = new java.io.FileInputStream(scriptName);
	    parse(file);
	} catch (java.io.FileNotFoundException ex) {
	    ex.printStackTrace();
	    System.exit(10);
	}
   }
   
   public RunSQLScript() {
      parse(System.in);
   }
   
   public static void main(String args[]) {
       if (args.length == 1) { 
	   new RunSQLScript(args[0]);
       } else {
	   new RunSQLScript();
       }
   }
   
   private void parse(InputStream script) {
      try {
         Class.forName(DRIVER).newInstance();
         
         String query = "";
         Connection conn = null;
         Statement s = null;
         
         String line;
         int lineNum = 0;
         BufferedReader br = new BufferedReader(new InputStreamReader(script));
         while ((line = br.readLine()) != null) {
            if (lineNum == 0) {
               // Get connection and create statement
               conn = DriverManager.getConnection(line);
               conn.setAutoCommit(true);
               s = conn.createStatement();               
               lineNum++;
               continue;
            }            
            
            // Other lines            
            if (line.indexOf(";") == -1) {
               // Add line to query until it reaches the end
               query += line;
            }
            else if (line.startsWith("jdbc")) {
               // Last line - Disconnect statement
               try {
                  DriverManager.getConnection(line);
               }
               catch (SQLException ex) {                  
               }
            }
            else {
               query += line;
               // Strip off ; at the end
               query = query.substring(0, query.length()-1);
               // Execute the query
               if (s.execute(query)) {
            	   do {
	            	   ResultSet rs = s.getResultSet();
	            	   ResultSetMetaData rmd = rs.getMetaData();
	        		   for (int i=1; i<=rmd.getColumnCount(); ++i) {
	        			   System.out.print(rmd.getColumnLabel(i) + "\t");
	        		   }
	        		   System.out.println();
	            	   while (rs.next()) {
	            		   for (int i=1; i<=rmd.getColumnCount(); ++i) {
	            			   System.out.printf("%s\t", rs.getObject(i) == null ? "(null)" : rs.getObject(i).toString());
	            		   }
	            		   System.out.println();
	            	   }
            	   } while (s.getMoreResults() && s.getUpdateCount() == -1);
               }
               // Reset query
               query = "";
            }
               
            lineNum++;
         }
         
         s.close();
         conn.close();
      }
      catch (Exception ex) {
         ex.printStackTrace();
	 System.exit(10);
      }
      System.exit(0);
   }
   
   private void test() {  
      try {
         Connection conn = DriverManager.getConnection(
              "jdbc:derby:databases/DbAsset-NA;bootPassword=a1x24i9o15m13a1;");
         Statement s = conn.createStatement();
         
         String query = "SELECT * FROM ASSET";
         
         ResultSet rs = s.executeQuery(query);
         while (rs.next()) {
            System.out.println("axioma id is " +  rs.getString("axioma_id"));
            System.out.println("adv_20d is " +  rs.getDouble("adv_20d"));
            System.out.println("retrun_20d is " +  rs.getDouble("return_20d"));
         }
      }
      catch (SQLException ex) {
         ex.printStackTrace();
      }      
   }
}
