import java.io.*;  
import java.net.*;  
public class MyServer {  
	public static void main(String[] args){  
		try{  
			byte[] data = new byte[1024];

			ServerSocket ss=new ServerSocket(6666);  
			Socket s=ss.accept();//establishes connection   

			int count = s.getInputStream()
					.read(data, 0, 1024);

			System.out.println("Receiving video...");
			File video = new File("ski2_copy.mp4");
			FileOutputStream fos = new FileOutputStream(video);

			while (count != -1) {
				fos.write(data, 0, count);
				count = s.getInputStream()
						.read(data, 0, 1024);
			}

			//DataInputStream dis=new DataInputStream(s.getInputStream());  
			//String  str=(String)dis.readUTF();  
			//System.out.println("message= "+str);  
			ss.close();  
			fos.close();
			s.close();
			System.out.println("Done receiving");

		}catch(Exception e){System.out.println(e);}  
	}  
}  


