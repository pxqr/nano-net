using System;

using Nanon.Data;
using Nanon.Math.Activator;
using Nanon.Math.Linear;
using Nanon.NeuralNetworks;
using Nanon.NeuralNetworks.Layer;
using Nanon.NeuralNetworks.Layer.Composition;


namespace Nanon.Test
{
	public class MergerSplitterTest
	{
		public static NeuralNetwork<Vector> Test(IDataSet<Vector, Vector> dataSet)
		{
			var hidden = 20;
			var count  = 10;
			
			
			var a = new ISingleLayer<Vector,Vector>[count];
			for (var i = 0; i < count; ++i)
				a[i] = new FullyConnectedLayer(dataSet.FirstInput.Size, hidden, new Tanh());
			
			var b = new ISingleLayer<Vector,Vector>[count];
			for (var i = 0; i < count; ++i)
				b[i] = new FullyConnectedLayer(hidden, dataSet.FirstOutput.Size / count, new Tanh());
			
			var splitter = new Splitter<Vector, Vector>(a);
			var merger   = new VectorMerger<Vector>(b);
			
			
			var comp = CompositeLayer<Vector, Vector[], Vector>.Compose2(splitter, merger);
			
			return new NeuralNetwork<Vector>(comp);
		}
	}
}

