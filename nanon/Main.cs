using System;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;

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
		
		static DataSet<Vector, Vector> LoadDataSet()
		{
			//var imagesPath = "/home/redner/projects/nanon/data/train-images.data"; //args[0];
			//var labelsPath = "/home/redner/projects/nanon/data/train-labels.data"; //args[1];	
			var imagesPath = "/home/redner/projects/nanon/data/test-images.data"; //args[0];
			var labelsPath = "/home/redner/projects/nanon/data/test-labels.data"; //args[1];	
			
			Console.WriteLine("Load data");
			var dataSet   = DataSet<Matrix, Matrix>.FromFile(imagesPath, labelsPath)
				           .Convert(x => x.ToVector, 
				                    x => Vector.FromIndex((int)x.Cells[0], 10));
			
			Console.WriteLine("Normalize data");
			var inputsNormalizator = new Normalization(dataSet.Inputs.ToArray());
			dataSet.TransformInputs(inputsNormalizator.Normalize);
			
			return dataSet;
		}
		
		static void Test(NeuralNetwork<Vector,Vector> network, IDataSet<Vector, Vector> dataSet)
		{
			var rtester = new HypothesisTester<Vector, Vector>(network);
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
			                //,   new List<int> { 10 }
							);
			
			var optimizer = new GradientDescent<Vector, Vector>(8, .001, x => 1, 5);
			var trainer   = new Trainer<Vector, Vector>(optimizer);
			
			Console.WriteLine("Initial");
			Test(network, dataSet);
			Console.WriteLine("StartLearning");
			
			Console.Write("Check Gradients");
			//Console.WriteLine(CheckModel(network, dataSet.FirstInput, dataSet.FirstOutput));
			
			for (int i = 0; i < 10; ++i)
			{
				trainer.Train(network, dataSet);
				Test(network, dataSet);
			}
			
			Console.WriteLine("EndLearning");
			Test(network, dataSet);
		}
	}
}
