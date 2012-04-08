using System;

using Nanon.NeuralNetworks.Layer;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public class Spreader<InputT, OutputT> : ISingleLayer<InputT, OutputT[]>
	{
		ISingleLayer<InputT, OutputT>[] layers;
		
		public Spreader()
		{
		}

		#region ISingleLayer[InputT,OutputT[]] implementation
		public OutputT[] FeedForward (InputT input)
		{
			throw new NotImplementedException ();
		}

		public InputT PropagateBackward (InputT input, InputT predSignal, OutputT[] error)
		{
			throw new NotImplementedException ();
		}

		public void FindGradient (InputT input, OutputT[] outputError)
		{
			var layersCount = layers.Length;
			
			if (outputError.Length != layersCount)
				throw new ArgumentException("layers and error count dont match");
			
			for (var i = 0; i < layersCount; ++i)
				layers[i].FindGradient(input, outputError[i]);
		}

		public Nanon.Math.Linear.Vector Gradient ()
		{
			throw new NotImplementedException ();
		}

		public void Correct (Nanon.Math.Linear.Vector gradients)
		{
			throw new NotImplementedException ();
		}

		public Nanon.Math.Linear.Vector ZeroGradients ()
		{
			throw new NotImplementedException ();
		}

		public OutputT[] Signal {
			get {
				throw new NotImplementedException ();
			}
		}

		public OutputT[] Output {
			get {
				throw new NotImplementedException ();
			}
		}
		#endregion
	}
}

