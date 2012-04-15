using System;

using Nanon.Math;
using Nanon.Math.Linear;
using Nanon.Model;
using Nanon.Model.Regression;
using Nanon.NeuralNetworks;
using Nanon.NeuralNetworks.Layer;
using Nanon.NeuralNetworks.Layer.Composition;
using Nanon.Math.Activator;

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

		public void Gradient (InputT input, Vector trainingOutput)
		{
			var output = layers.FeedForward(input);
			var error  = trainingOutput.ZeroCopy();
			// o - t
			output.Sub(trainingOutput, error);
			
			// f'(o)
			//var deriv = output.ZeroCopy();
			//output.Transform(new Tanh().Derivative, deriv);
			
			// f'(o) * (o - t)
			//error.Mul(deriv, error);
			
			// cancel backprop if isnt neccessary
			if (error.EuclideanNorm < 0.1d)
			{
				++counter;
				return;
			}
			
			layers.Backprop(input, error);
		}
		
		public static int counter = 0;
		
		public void Correct (double coeff)
		{
			layers.Correct(coeff);
		}
		
		#endregion
	}
}

