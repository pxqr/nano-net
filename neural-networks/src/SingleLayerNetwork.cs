using System.Collections.Generic;

using Nanon.Math.Linear;
using Nanon.Model;
using Nanon.Model.Regression;
using Nanon.NeuralNetworks.Layer;

namespace Nanon.NeuralNetworks
{
	public class SingleLayerNetwork<InputT> : IRegression<InputT, Vector>, IHypothesis<InputT, Vector>
	{
		ISingleLayer<InputT, Vector> layer;
		
		public SingleLayerNetwork(ISingleLayer<InputT, Vector> layerA)
		{
			layer = layerA;
		}
		
		#region IRegression[InputT,Vector] implementation
		
		public Vector Predict(InputT input)
		{
			return layer.FeedForward(input);	
		}
		
		#endregion		
		
		#region IHypothesis[InputT,OutputT] implementation
		
		public double Cost(InputT input, Vector output)
		{
			var prediction = layer.FeedForward(input);
			var iftrue     = prediction.Map(System.Math.Log) * output;
			var iffals     = prediction.Map(x => System.Math.Log(0.01 + (1 - x))) * output.Map(x => 1 - x);
			return - (iftrue + iffals);
		}

		public Vector[] Gradient(InputT input, Vector output)
		{
			var prediction = layer.FeedForward(input);
			var error      = output - prediction;
			
			layer.FindGradient(input, error);

			return new Vector[] { 
				//NumericalGradient(input, output)
				layer.Gradient() 
			};
		}

		public void Correct(Vector[] gradient)
		{
			layer.Correct(gradient[0]);
		}

		public Vector[] ZeroGradient 
		{
			get 
			{
				return new Vector[] { layer.ZeroGradients() };
			}
		}
		
		#endregion
		
		// diagnostic
		/*
		public Vector NumericalGradient(InputT input, Vector output)
		{
			var weightsCount = layer.WeightsSize;
			var grad = new Vector(weightsCount);
			var epsilon = 0.00001d;
			
			for (var i = 0; i < weightsCount; ++i)
			{
				var old = layer[i];
				
				layer[i] = old - epsilon;
				var lossDown = Cost(input, output);
				
				layer[i] = old + epsilon;
				var lossUp   = Cost(input, output);
				
				grad[i] = (lossUp - lossDown) / (2 * epsilon);
				
				layer[i] = old;
			}
			
			return grad.ToVector;
		}
		*/
	}
}