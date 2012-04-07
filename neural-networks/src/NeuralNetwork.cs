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
	public class NeuralNetwork<InputT, OutputT> : IRegression<InputT, OutputT>, IHypothesis<InputT, OutputT>
	{
		ICompositeLayer<InputT, OutputT> layers;
		Func<OutputT, OutputT, double>   costFunction;
		Func<OutputT, OutputT, OutputT>  errorFunction;
		
		public NeuralNetwork(ICompositeLayer<InputT, OutputT> layersA,
			 					 Func<OutputT, OutputT, double>   costFunctionA,
		                         Func<OutputT, OutputT, OutputT>  errorFunctionA)
		{
			layers        = layersA;
			costFunction  = costFunctionA;
			errorFunction = errorFunctionA;
		}
		
		
		#region IRegression[InputT,OutputT] implementation
		
		public OutputT Predict (InputT input)
		{
			return layers.FeedForward(input);
		}
		
		#endregion

		#region IHypothesis[InputT,OutputT] implementation
		
		public double Cost (InputT input, OutputT output)
		{
			var prediction = layers.FeedForward(input);
			return costFunction(prediction, output);
		}

		public Vector[] Gradient (InputT input, OutputT output)
		{
			var prediction = layers.FeedForward(input);
			var error      = errorFunction(output, prediction);
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

