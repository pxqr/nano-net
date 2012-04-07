using System;
using System.IO;

namespace Nanon.Converter
{
	class MainClass
	{
		static string helpFilePath = "help.txt";
			
		static void ShowHelp ()
		{
			try {
				using (var sr = new StreamReader(helpFilePath)) {
					String line;
					while ((line = sr.ReadLine()) != null) 
						Console.WriteLine (line);        
				}
			} catch (Exception e) {
				Console.WriteLine ("The help file `" + helpFilePath + 
                "' could be read");
				Console.Write (e.Message);
			}
		}
    
		public static void  ConvertFile (string inputPath, string ouputPath)
		{
			Console.WriteLine("Trying to open input.");
			
			try
			{
				using (var input = new BinaryReader(File.Open(inputPath, FileMode.Open))) {
				
					try
					{
						Console.WriteLine("Trying to create output.");
						using (var output = new BinaryWriter(File.Open(ouputPath, FileMode.OpenOrCreate))) {
							
							try
							{
								Console.WriteLine("Trying to convert.");
								Converter.Convert (input, output);
							} catch (Exception e) {
								Console.WriteLine(">> Cant convert.");
								Console.WriteLine(">> Exit with error: {0}", e.Message);		
							}
						}
					} catch (Exception e) {
			  			Console.WriteLine(">> Unable to create `{0}'", ouputPath);
						Console.WriteLine(">> Exit with error: {0}", e.Message);		
						return;
					}
				}
			} catch (Exception e) {
			  	Console.WriteLine(">> Unable to open `{0}'", inputPath);
				Console.WriteLine(">> Exit with error: {0}", e.Message);		
				return;
			}
		}
		
		public static void Main (string[] args)
		{
			if (args.Length != 2) {
				ShowHelp ();
				return;
			}
      
			var inputPath =  args [0];
			var outputPath = args [1];
			
			Console.WriteLine("Convert from `{0}' -> `{1}'",
			                  inputPath, outputPath);
			
			ConvertFile (inputPath, outputPath);
		}
	}
}
