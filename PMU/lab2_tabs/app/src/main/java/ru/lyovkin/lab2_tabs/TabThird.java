package ru.lyovkin.lab2_tabs;

import android.os.Bundle;

import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import java.lang.reflect.Constructor;

public class TabThird extends Fragment {

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
//        return inflater.inflate(R.layout.fragment_tab_third, container, false);

        TableLayout table = new TableLayout(getActivity());

        for (int i = 1; i <= 4; i++) {
            TableRow row = new TableRow(getActivity());
            for (int j = 1; j <= 4; j++) {
                Button tv = new Button(getActivity());
                tv.setText(String.valueOf(i * j));
                row.addView(tv);
            }
            table.addView(row);
        }
        return table;
    }
}