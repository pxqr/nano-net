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
		static string trainImagesPath = "/home/redner/projects/nanon/data/train-images.data"; 
		static string trainLabelsPath = "/home/redner/projects/nanon/data/train-labels.data"; 	
		static string testImagesPath  = "/home/redner/projects/nanon/data/test-images.data"; 
		static string testLabelsPath  = "/home/redner/projects/nanon/data/test-labels.data"; 
	
		public static int Main(string[] args)
		{
			CNNTest.Test(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath);

			return 0;
		}
	}
}
