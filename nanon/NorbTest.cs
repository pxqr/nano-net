using System;
using Nanon.Data;
using Nanon.Math.Linear;
using Nanon.Model;
using System.Linq;
using Nanon.Learning.Tools;
using Nanon.Model.Classifier;
using Nanon.NeuralNetworks;
using System.Diagnostics;
using Nanon.Learning.Optimization;

namespace Nanon.Test
{
	public class NorbTest
	{
		static DataSet<Matrix, Vector> trainDataSet;
		static DataSet<Matrix, Vector> testDataSet;
		
		static DataSet<Matrix, Vector> Load(string trainImagesPath, string trainLabelsPath)
		{
			Console.WriteLine("Load data from {0} \n{1}", trainImagesPath, trainLabelsPath);
			return DataSet<Matrix, Matrix>.FromFile(trainImagesPath, trainLabelsPath)
				   				          .Convert(x => x, 
				                 				   x => Vector.FromIndex((int)x.Cells[0], 5, -0.8d, 0.8d));
		}
		
		static void LoadDataSet(string trainImagesPath, string trainLabelsPath)
		{
			var dataSet = Load(trainImagesPath, trainLabelsPath).Take(9500);
			GC.Collect();
			
			Console.WriteLine("Normalize data");
			Vector[] vectors = dataSet.Inputs.Select(x => x.ToVector).ToArray();
			var inputsNormalizator = new Normalization(vectors);
			
			foreach (var input in dataSet.Inputs)
				inputsNormalizator.Normalize(input.ToVector);
			
			trainDataSet = dataSet.Take(0, 8000);
			testDataSet  = dataSet.Take(8000, 9500);
		}
		
		static double Test(IHypothesis<Matrix, Vector> network, IDataSet<Matrix, Vector> dataSet, double oldcost = Double.PositiveInfinity)
		{
			var rtester = new RegressionTester<Matrix>(network);
			var cost = rtester.Test(dataSet);
			    
			var classifier = new MaxFitClassifier<Matrix>(network);				
			var ctester = new ClassifierTester<Matrix, Vector, int>(classifier, x => x.IndexOfMax);
			var accuracy = ctester.Test(dataSet);
				
			Console.Write("C {0} | {1}%", cost, accuracy * 100);
			
			if (oldcost < cost)
			{
				Console.WriteLine();
				Console.WriteLine("Warning: probably weights will divergent!");
			}
			
			return cost;
		}
		
		public static void Test(string trainImagesPath, string trainLabelsPath)
		{
			LoadDataSet(trainImagesPath, trainLabelsPath);
			GC.Collect();
				
			var network   = NetworkBuilder.CreateNorb(trainDataSet);
			
			Console.WriteLine("Initial");
			Test(network, trainDataSet);
			Console.WriteLine("StartLearning");
			
			var cost = Double.PositiveInfinity;
			var timer = new Stopwatch();
			timer.Start();				
			
			var optimizer = new GradientDescent<Matrix, Vector>(4, .01, 1, 
			    x => { 
					timer.Stop();		
					Console.Write("Ignored {0}% of samples ", 100 * NeuralNetwork<Matrix>.counter / (double)trainDataSet.Inputs.Count());
					Console.WriteLine("and gradient descent step time: {0} ms", timer.ElapsedMilliseconds);	
					NeuralNetwork<Matrix>.counter = 0;
					Console.Write("trainSet: ");
					cost = Test(x, trainDataSet, cost);
					Console.Write("testSet: ");
					Test(x, testDataSet);
					Console.WriteLine();
					timer.Reset();
					timer.Start();				
				});
			
			var trainer   = new Trainer<Matrix, Vector>(optimizer);
			
			for (var i = 0; i < 5; ++i)
			{
				Console.WriteLine("Generation {0}", i);	
				trainer.Train(network, trainDataSet);
				optimizer.IterationCount  += 2;
				//optimizer.LearningRate    += 0.01;
				optimizer.InitialStepSize *= 2;
			}
			
			Console.WriteLine("EndLearning");
		}
	}
}