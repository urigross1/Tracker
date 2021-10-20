import java.io.*;  
import java.net.*;  



public class MyClient {  
	public static void main(String[] args) {  
		try{      
		
			System.out.println("Connecting...");
			InputStream is = new FileInputStream(new File("ski2.mp4"));
			byte[] bytes = new byte[1024];

			Socket s = new Socket("localhost",6666);  

			OutputStream stream = s.getOutputStream();
			int count = is.read(bytes, 0, 1024);
			while (count != -1) {
				stream.write(bytes, 0, 1024);
			 
				count = is.read(bytes, 0, 1024);
			}


			//DataOutputStream dout=new DataOutputStream(stream);  
			//dout.writeUTF("Hello Server");  
			//dout.flush();  
			//dout.close();  

			is.close();
			stream.close();

			s.close();  
			System.out.println("2");

		}catch(Exception e){System.out.println(e);}  
	}  
}  





