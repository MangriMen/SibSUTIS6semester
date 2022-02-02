package ru.lyovkin.lab2_tabs;

import android.graphics.pdf.PdfDocument;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Adapter;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentPagerAdapter;
import androidx.lifecycle.Lifecycle;
import androidx.lifecycle.LifecycleObserver;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;

public class MainActivity extends AppCompatActivity {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        PagerAdapter pagerAdapter = new PagerAdapter(getSupportFragmentManager(), new Lifecycle() {
            @Override
            public void addObserver(@NonNull LifecycleObserver observer) {

            }

            @Override
            public void removeObserver(@NonNull LifecycleObserver observer) {

            }

            @NonNull
            @Override
            public State getCurrentState() {
                return State.CREATED;
            }
        });

        ViewPager2 viewPager = (ViewPager2) findViewById(R.id.vpMain);
        viewPager.setAdapter(pagerAdapter);

        TabLayout tlMain = (TabLayout) findViewById(R.id.thMain);

        TabLayoutMediator tabLayoutMediator = new TabLayoutMediator(tlMain, viewPager, (tab, position) -> {
            tab.setText("Вкладка " + position);
        });
        tabLayoutMediator.attach();
    }

    private static class PagerAdapter extends FragmentStateAdapter {
        public PagerAdapter(@NonNull FragmentManager fragmentManager, @NonNull Lifecycle lifecycle) {
            super(fragmentManager, lifecycle);
        }

        @Override
        public int getItemCount() {
            return 3;
        }

        @NonNull
        @Override
        public Fragment createFragment(int position) {
            switch (position) {
                case 0:
                    return new TabFirst();
                case 1:
                    return new TabSecond();
                case 2:
                    return new TabThird();
            }
            return new Fragment();
        }
    }
}