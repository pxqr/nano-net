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
		static string testImagesPath = "/home/redner/projects/nanon/data/test-images.data"; 
		static string testLabelsPath = "/home/redner/projects/nanon/data/test-labels.data"; 
		static DataSet<Vector, Vector> testDataSet;
		
		static DataSet<Vector, Vector> Load(string trainImagesPath, string trainLabelsPath)
		{
			Console.WriteLine("Load data from {0} \n{1}", trainImagesPath, trainLabelsPath);
			return DataSet<Matrix, Matrix>.FromFile(trainImagesPath, trainLabelsPath)
				   				          .Convert(x => x.ToVector, 
				                 				   x => Vector.FromIndex((int)x.Cells[0], 10, -0.2d, 0.2d));
		}
		
		static DataSet<Vector, Vector> LoadDataSet()
		{
			var trainDataSet   = Load(trainImagesPath, trainLabelsPath);
			testDataSet    = Load (testImagesPath, testLabelsPath);
			
			Console.WriteLine("Normalize data");
			var inputsNormalizator = new Normalization(trainDataSet.Inputs.ToArray());
			trainDataSet.TransformInputs(inputsNormalizator.Normalize);
			testDataSet.TransformInputs(inputsNormalizator.Normalize);
			
			return trainDataSet;
		}
		
		static void Test(NeuralNetwork<Vector> network, IDataSet<Vector, Vector> dataSet)
		{
			var rtester = new RegressionTester<Vector>(network);
			var cost = rtester.Test(dataSet);
			    
			var classifier = new MaxFitClassifier<Vector>(network);				
			var ctester = new ClassifierTester<Vector, Vector, int>(classifier, x => x.IndexOfMax);
			var accuracy = ctester.Test(dataSet);
				
			Console.WriteLine("cost {0}, accuracy {1}%", cost, accuracy * 100);
		}
		
		
		/*
		static bool CheckModel(SingleLayerNetwork<Vector> network, Vector input, Vector output)
		{
			var numgrad   = network.NumericalGradient(input, output);
			var modelgrad = network.Gradient(input, output);
			var diff = (numgrad - modelgrad[0]);
				
			return diff.EuclideanNorm < 0.01;
		}
		*/
		
		public static void Main (string[] args)
		{				
			var dataSet   = LoadDataSet();
			var network   = NetworkBuilder.Create(dataSet
			                , new Tanh()
			                //, new List<int> { 50 }
							);
			
			var optimizer = new GradientDescent<Vector, Vector>(1, .005, x => 1, 5);
			var trainer   = new Trainer<Vector, Vector>(optimizer);
			
			
			Console.WriteLine("Initial");
			Test(network, testDataSet);
			Console.WriteLine("StartLearning");
			
			for (int i = 0; i < 10; ++i)
			{
				trainer.Train(network, dataSet);
				optimizer.Momentum = optimizer.Momentum / 2;
				optimizer.InitialStepSize += 1;
				optimizer.IterationCount += 2;
				
				Console.Write("train set: ");
				Test(network, dataSet);
				Console.Write("test set: ");
				Test(network, testDataSet);
			}
			
			Console.WriteLine("EndLearning");
			Test(network, testDataSet);
		}
	}
}
