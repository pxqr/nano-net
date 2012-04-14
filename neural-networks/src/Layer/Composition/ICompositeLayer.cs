using System;
using System.Collections.Generic;

using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public interface ICompositeLayer<InputT, OutputT>
	{
		OutputT FeedForward(InputT input);
		
		void Backprop(InputT input, OutputT error);
		InputT  PropagateBackward(InputT input, OutputT error);
		
		void   Correct(double gradients);
	}
}

