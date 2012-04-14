using System;
using System.Collections.Generic;

using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer
{
	// Layer :: input -> output
	public interface ISingleLayer<InputT, OutputT>
	{
		OutputT FeedForward(InputT input);
		
		InputT  PropagateBackward(InputT input, OutputT error);
		OutputT Output { get; }
		
		void Gradient(InputT input, OutputT outputError);		
		void Correct(double gradients);
	}
}