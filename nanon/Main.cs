using System;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;

using Nanon.Math.Activator;
using Nanon.Data;
using Nanon.Learning.Optimization;
using Nanon.Learning.Tools;
using Nanon.Math.Linear;
using Nanon.Model.Classifier;
using Nanon.Model;
using Nanon.Statistics.Linear;
using Nanon.Statistics.Logistic;
using Nanon.NeuralNetworks;
using Nanon.NeuralNetworks.Layer;
using Nanon.Math.Series;


namespace Nanon.Test
{
	class MainClass
	{
		static string trainImagesPath = "/home/redner/projects/nanon/data/mnist/converted/train-images.data"; 
		static string trainLabelsPath = "/home/redner/projects/nanon/data/mnist/converted/train-labels.data"; 	
		static string testImagesPath  = "/home/redner/projects/nanon/data/mnist/converted/test-images.data"; 
		static string testLabelsPath  = "/home/redner/projects/nanon/data/mnist/converted/test-labels.data"; 
		
		static string norbTrainImagesPath = "/home/redner/projects/nanon/data/snorb/trunk/train-trunk-10000.data"; 
		static string norbTrainLabelsPath = "/home/redner/projects/nanon/data/snorb/trunk/train-labels.data"; 	
	
		public static int Main(string[] args)
		{
			//MatrixTest.Test();
			CNNTest.Test(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath);
			//NorbTest.Test(norbTrainImagesPath, norbTrainLabelsPath);
			return 0;
		}
	}
}
