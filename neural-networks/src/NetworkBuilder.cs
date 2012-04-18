using System;
using System.Collections.Generic;

using Nanon.Math.Activator;
using Nanon.Math.Linear;
using Nanon.Data;
using Nanon.NeuralNetworks.Layer;
using Nanon.NeuralNetworks.Layer.Composition;
using Nanon.NeuralNetworks.Layer.Convolutional;

namespace Nanon.NeuralNetworks
{
	public class NetworkBuilder
	{
		// single layer network
		public static NeuralNetwork<Vector> Create(IDataSet<Vector, Vector> dataSet, IActivator activator)
		{
			var workLayer = new FullyConnectedLayer(dataSet.FirstInput.Size, dataSet.FirstOutput.Size, activator);
			var outputLayer = new OutputLayer<Vector>();
			var layers = new CompositeLayer<Vector, Vector, Vector>(workLayer, outputLayer);
			return new NeuralNetwork<Vector>(layers);
		}
		
		public static NeuralNetwork<Vector> Create(IDataSet<Vector, Vector> dataSet, IActivator activator, List<int> hiddenSizes)
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
			
			return new NeuralNetwork<Vector>(compositeLayer);
		}
		
		public static NeuralNetwork<Matrix> Create(IDataSet<Matrix, Vector> dataSet)
		{
			var count  = 5;
			
			var a = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				a[i] = new MatrixConvolutor(28, 28, 24, 24, new Tanh());
			
			var b = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				b[i] = new MatrixSubsampler(24, 24, 12, 12, new Tanh());
			
			var c = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				c[i] = new MatrixConvolutor(12, 12, 8, 8, new Tanh());
			
			var d = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				d[i] = new MatrixSubsampler(8, 8, 4, 4, new Tanh());
			
			var splitter    = new Splitter<Matrix, Matrix>(a);
			var applicator1 = new Applicator<Matrix, Matrix>(b);
			var applicator2 = new Applicator<Matrix, Matrix>(c);
			var merger      = new MatrixMerger<Matrix>(d);
			
			var classif  = new FullyConnectedLayer(16 * count, 10, new Tanh());
			
			var comp = CompositeLayer<Vector, Vector[], Vector>.Compose(splitter, 
			                                                            applicator1,
			                                                            applicator2,
			                                                            merger,
			                                                            classif);
			
			return new NeuralNetwork<Matrix>(comp);
		}
		
		public static NeuralNetwork<Matrix> CreateSemi(IDataSet<Matrix, Vector> dataSet)
		{
			var count  = 10;
			
			var a = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				a[i] = new MatrixConvolutor(28, 28, 24, 24, new Tanh());
			
			var b = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				b[i] = new MatrixSubsampler(24, 24, 12, 12, new Tanh());
			
			var splitter    = new Splitter<Matrix, Matrix>(a);
			var merger      = new MatrixMerger<Matrix>(b);
				
			var classif  = new FullyConnectedLayer(144 * count, 50, new Tanh());
			var classif2 = new FullyConnectedLayer(50, 10, new Tanh());
			
			var comp = CompositeLayer<Vector, Vector[], Vector>.Compose(splitter, 
			                                                            merger,
			                                                            classif,
			                                                            classif2);
			
			return new NeuralNetwork<Matrix>(comp);
		}
		
		public static NeuralNetwork<Matrix> CreateMnist(IDataSet<Matrix, Vector> dataSet)
		{
			var count  = 10;
			
			var a = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				a[i] = new MatrixConvolutor(28, 28, 24, 24, new Tanh());
			
			var b = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				b[i] = new MatrixSubsampler(24, 24, 12, 12, new Tanh());
			
			var splitter    = new Splitter<Matrix, Matrix>(a);
			var merger      = new MatrixMerger<Matrix>(b);
				
			var classif  = new FullyConnectedLayer(144 * count, 50, new Tanh());
			var classif2 = new FullyConnectedLayer(50, 10, new Tanh());
			
			var comp = CompositeLayer<Vector, Vector[], Vector>.Compose(splitter, 
			                                                            merger,
			                                                            classif,
			                                                            classif2);
			
			return new NeuralNetwork<Matrix>(comp);
		}
		
		public static NeuralNetwork<Matrix> CreateConv(IDataSet<Matrix, Vector> dataSet)
		{
			var count  = 5;
			var branch = 5;
			
			var a = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				a[i] = new MatrixConvolutor(28, 28, 24, 24, new Tanh());
			
			var b = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				b[i] = new MatrixSubsampler(24, 24, 12, 12, new Tanh());
			
			var c = new ISingleLayer<Matrix, Matrix>[branch];
			for (var i = 0; i < branch; ++i)
				c[i] = new MatrixConvolutor(12, 12, 8, 8, new Tanh());
			
			var d = new ISingleLayer<Matrix, Matrix>[count * branch];
			for (var i = 0; i < count * branch; ++i)
				d[i] = new MatrixSubsampler(8, 8, 4, 4, new Tanh());
			
			var splitter    = new Splitter<Matrix, Matrix>(a);
			var applicator1 = new Applicator<Matrix, Matrix>(b);
			var applicator2 = new Combiner<Matrix, Matrix>(c, count);
			var merger      = new MatrixMerger<Matrix>(d);
			
			var classif  = new FullyConnectedLayer(16 * branch * count, 10, new Tanh());
			
			var comp = CompositeLayer<Vector, Vector[], Vector>.Compose(splitter, 
			                                                            applicator1,
			                                                            applicator2,
			                                                            merger,
			                                                            classif);
			
			return new NeuralNetwork<Matrix>(comp);
		}
		
		public static NeuralNetwork<Matrix> CreateNorb(IDataSet<Matrix, Vector> dataSet)
		{
			var count  = 10;
			
			var a = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				a[i] = new MatrixConvolutor(96, 96, 80, 80, new Tanh());
			
			var b = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				b[i] = new MatrixSubsampler(80, 80, 40, 40, new Tanh());
			
			var c = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				c[i] = new MatrixConvolutor(40, 40, 30, 30, new Tanh());
			
			var d = new ISingleLayer<Matrix, Matrix>[count];
			for (var i = 0; i < count; ++i)
				d[i] = new MatrixSubsampler(30, 30, 15, 15, new Tanh());
			
			var splitter    = new Splitter<Matrix, Matrix>(a);
			var applicator1 = new Applicator<Matrix, Matrix>(b);
			var applicator2 = new Applicator<Matrix, Matrix>(c);
			var merger      = new MatrixMerger<Matrix>(d);
			
			var classif  = new FullyConnectedLayer(225 * count, 5, new Tanh());
			
			var comp = CompositeLayer<Vector, Vector[], Vector>.Compose(splitter, 
			                                                            applicator1, 
			                                                            applicator2,
			                                                            merger,
			                                                            classif
			                                                           );
			
			return new NeuralNetwork<Matrix>(comp);
		}
	}
}

