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


namespace Nanon
{
	class MainClass
	{
		static string trainImagesPath = "/home/redner/projects/nanon/data/train-images.data"; 
		static string trainLabelsPath = "/home/redner/projects/nanon/data/train-labels.data"; 	
		static string testImagesPath  = "/home/redner/projects/nanon/data/test-images.data"; 
		static string testLabelsPath  = "/home/redner/projects/nanon/data/test-labels.data"; 
		static DataSet<Vector, Vector> testDataSet;
		
		static DataSet<Vector, Vector> Load(string trainImagesPath, string trainLabelsPath)
		{
			Console.WriteLine("Load data from {0} \n{1}", trainImagesPath, trainLabelsPath);
			return DataSet<Matrix, Matrix>.FromFile(trainImagesPath, trainLabelsPath)
				   				          .Convert(x => x.ToVector, 
				                 				   x => Vector.FromIndex((int)x.Cells[0], 10, -1.0d, 1.0d));
		}
		
		static DataSet<Vector, Vector> LoadDataSet()
		{
			var trainDataSet   = Load(trainImagesPath, trainLabelsPath).Take (2000);
			testDataSet    = Load (testImagesPath, testLabelsPath).Take (2000);
			
			Console.WriteLine("Normalize data");
			var inputsNormalizator = new Normalization(trainDataSet.Inputs.ToArray());
			trainDataSet.TransformInputs(inputsNormalizator.Normalize);
			testDataSet.TransformInputs(inputsNormalizator.Normalize);
			
			return trainDataSet;
		}
		
		static double Test(IHypothesis<Vector, Vector> network, IDataSet<Vector, Vector> dataSet, double oldcost = Double.PositiveInfinity)
		{
			var rtester = new RegressionTester<Vector>(network);
			var cost = rtester.Test(dataSet);
			    
			var classifier = new MaxFitClassifier<Vector>(network);				
			var ctester = new ClassifierTester<Vector, Vector, int>(classifier, x => x.IndexOfMax);
			var accuracy = ctester.Test(dataSet);
				
			Console.Write("C {0} | {1}%", cost, accuracy * 100);
			
			if (oldcost < cost)
			{
				Console.WriteLine();
				Console.WriteLine("Warning: probably weights will divergent!");
			}
			
			return cost;
		}
		
		public static void Main (string[] args)
		{			
			var dataSet   = LoadDataSet();
			var network   = Nanon.Test.MergerSplitterTest.TestVectorSubsampler(dataSet);
							/*NetworkBuilder.Create(dataSet
			                , new Tanh()
			                //, new List<int> { 20 }
							);*/

			var cost = Double.PositiveInfinity;
			var optimizer = new GradientDescent<Vector, Vector>(10, 1, x => 1, 5, 
			    x => { 
					Console.Write("trainSet: ");
					cost = Test(x, dataSet, cost);
					Console.Write(" || ");
					Console.Write("testSet:  ");
					Test(x, testDataSet);
					Console.WriteLine();
				});
			
			var trainer   = new Trainer<Vector, Vector>(optimizer);
			
			
			Console.WriteLine("Initial");
			Test(network, dataSet);
			Console.WriteLine("StartLearning");
			trainer.Train(network, dataSet);
			Console.WriteLine("EndLearning");
			Test(network, testDataSet);
		}
	}
}
