using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Icycle : MonoBehaviour
{
    public float speed = 6f;
    public float speedIncrease = 0f;

    void Update()
    {
        speed += speedIncrease * Time.deltaTime;
        transform.Translate(Vector3.down * speed * Time.deltaTime);
    }
}
