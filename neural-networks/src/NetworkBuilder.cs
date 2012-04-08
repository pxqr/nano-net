using System;
using System.Collections.Generic;

using Nanon.Math.Activator;
using Nanon.Math.Linear;
using Nanon.Data;
using Nanon.NeuralNetworks.Layer;
using Nanon.NeuralNetworks.Layer.Composition;

namespace Nanon.NeuralNetworks
{
	public class NetworkBuilder
	{
		// single layer network
		public static NeuralNetwork<Vector, Vector> Create(IDataSet<Vector, Vector> dataSet, IActivator activator)
		{
			var workLayer = new FullyConnectedLayer(dataSet.FirstInput.Size, dataSet.FirstOutput.Size, activator);
			var outputLayer = new OutputLayer<Vector>();
			var layers = new CompositeLayer<Vector, Vector, Vector>(workLayer, outputLayer);
			return new NeuralNetwork<Vector, Vector>(layers, EuLossFunc, ErrorFunction);
		}
		
		public static NeuralNetwork<Vector, Vector> Create(IDataSet<Vector, Vector> dataSet, IActivator activator, List<int> hiddenSizes)
		{
			if (hiddenSizes.Count == 0)
				return Create(dataSet, activator);
			
			var inputSize  = dataSet.FirstInput.Size;
			var outputSize = dataSet.FirstOutput.Size;
			var sizes = new List<int>{inputSize};
			sizes.AddRange(hiddenSizes);
			sizes.Add(outputSize);
			
			var layerCount = sizes.Count - 1;
			var layers = new ISingleLayer<Vector, Vector>[layerCount];
			
			for (var i = 0; i < layerCount; ++i)
				layers[i] = new FullyConnectedLayer(sizes[i], sizes[i + 1], activator);
			
			var compositeLayer = LayerCompositor.ComposeGeteroneneous(layers);
			
			return new NeuralNetwork<Vector, Vector>(compositeLayer, EuLossFunc, ErrorFunction);
		}
		
		static Vector ErrorFunction(Vector output, Vector prediction)
		{
			return output - prediction;
		}
		
		static double CostFunction(Vector prediction, Vector output)
		{
			var iftrue     = prediction.Map(System.Math.Log) * output;
			var iffals     = prediction.Map(x => System.Math.Log(1 - x)) * output.Map(x => 1 - x);
			if (Double.IsNaN(iffals) || Double.IsNaN(iftrue))
				return 0;
			return - (iftrue + iffals);
		}
		
		static double EuLossFunc(Vector prediction, Vector output)
		{
			return (prediction - output).EuclideanNorm;
		}
	}
}

