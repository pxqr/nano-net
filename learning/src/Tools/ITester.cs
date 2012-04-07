using System;

using Nanon.Data;

namespace Nanon.Learning.Tools
{
	public interface ITester<InputT, OutputT>
	{
		double Test(IDataSet<InputT, OutputT> dataSet);
	}
}

