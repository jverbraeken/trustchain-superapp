package nl.tudelft.trustchain.distributedAI;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;

import androidx.fragment.app.Fragment;

public class DistributedFragmentJava extends Fragment {

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        return (FrameLayout) inflater.inflate(R.layout.activity_distributed, container, false);
    }


}
