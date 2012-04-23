using System;
using Nanon.Data;
using Nanon.Math.Linear;
using Nanon.Model;
using Nanon.Learning.Tools;
using Nanon.Model.Classifier;
using Nanon.Learning.Optimization;
using Nanon.NeuralNetworks;
using Nanon.Math.Activator;
using System.Linq;
using System.Collections.Generic;

namespace Nanon.Test
{
	public class SNNTest
	{
		static DataSet<Vector, Vector> testDataSet;
		static DataSet<Vector, Vector> trainDataSet;
		
		static DataSet<Vector, Vector> Load(string trainImagesPath, string trainLabelsPath)
		{
			Console.WriteLine("Load data from {0} \n{1}", trainImagesPath, trainLabelsPath);
			return DataSet<Matrix, Matrix>.FromFile(trainImagesPath, trainLabelsPath)
				   				          .Convert(x => x.ToVector, 
				                 				   x => Vector.FromIndex((int)x.Cells[0], 10, -1.0d, 1.0d));
		}
		
		static DataSet<Vector, Vector> LoadDataSet(string trainImagesPath, string trainLabelsPath, string testImagesPath, string testLabelsPath)
		{
			var trainDataSet   = Load(trainImagesPath, trainLabelsPath);
			GC.Collect();
			testDataSet    = Load (testImagesPath, testLabelsPath);
			GC.Collect();
			
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
		
		public static void Test(string trainImagesPath, string trainLabelsPath, string testImagesPath, string testLabelsPath)
		{			
			var dataSet   = LoadDataSet(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath);
			GC.Collect();
			
			var network   = NetworkBuilder.Create(dataSet
			                , new Tanh()
			                , new List<int> { 50 }
							);
			
			var cost = Double.PositiveInfinity;
			var optimizer = new GradientDescent<Vector, Vector>(5, 0.02, 2, 
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
			
			for (var i = 0; i < 3; ++i)
			{
				Console.WriteLine("next generation");	
				trainer.Train(network, dataSet.Set);
				optimizer.IterationCount += 1;
				optimizer.InitialStepSize *= 2;
			}
			
			Console.WriteLine("EndLearning");
			Test(network, testDataSet);
		}
	}
}

