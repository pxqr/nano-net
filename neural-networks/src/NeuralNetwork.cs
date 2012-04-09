using System;

using Nanon.Math;
using Nanon.Math.Linear;
using Nanon.Model;
using Nanon.Model.Regression;
using Nanon.NeuralNetworks;
using Nanon.NeuralNetworks.Layer;
using Nanon.NeuralNetworks.Layer.Composition;

namespace Nanon.NeuralNetworks
{
	public class NeuralNetwork<InputT> : IHypothesis<InputT, Vector>
	{
		ICompositeLayer<InputT, Vector> layers;
		
		public NeuralNetwork(ICompositeLayer<InputT, Vector> layersA)
		{
			layers        = layersA;
		}
		
		
		#region IRegression[InputT,OutputT] implementation
		
		public Vector Predict (InputT input)
		{
			return layers.FeedForward(input);
		}
		
		#endregion

		#region IHypothesis[InputT,OutputT] implementation

		public Vector[] Gradient (InputT input, Vector output)
		{
			var prediction = layers.FeedForward(input);
			var error      = output - prediction;
			layers.Backprop(input, error);
			return layers.Gradient().ToArray();
		}

		public void Correct (Vector[] gradient)
		{
			layers.Correct(ConsList<Vector>.FromArray(gradient));
		}

		public Nanon.Math.Linear.Vector[] ZeroGradient 
		{
			get 
			{
				return layers.ZeroGradients().ToArray();
			}
		}
		
		#endregion
	}
}

