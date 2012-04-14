using System;

using Nanon.Math.Activator;

namespace Nanon.NeuralNetworks.Layer
{
	public abstract class SingleLayer<InputT, OutputT> : ISingleLayer<InputT, OutputT>
	{
		protected IActivator activator;
		
		protected OutputT signals;
		protected OutputT outputs;
		
		public SingleLayer(IActivator activatorA)
		{
			activator = activatorA;
		}
		
		#region ISingleLayer[InputT,OutputT] implementation
		
		public static double OptimalInitEpsilon(int inputSize, int outputSize)
		{
			return System.Math.Sqrt(6) / System.Math.Sqrt(inputSize + outputSize);
		}
		
		public abstract OutputT FeedForward (InputT input);
		public abstract InputT PropagateBackward (InputT input, OutputT error);
		public abstract void Gradient (InputT input, OutputT outputError);
		public abstract void Correct (double gradients);

		public OutputT Signal {
			get {
				return signals;
			}
		}

		public OutputT Output {
			get {
				return outputs;
			}
		}
		
		#endregion
	}
}

