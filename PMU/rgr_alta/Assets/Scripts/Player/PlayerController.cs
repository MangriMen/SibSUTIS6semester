using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [SerializeField]
    private Vector2 gravity = new Vector2(-5f, 0f);

    private Rigidbody2D rb;
    
    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    void Update()
    {
        if (Input.GetButtonDown("Jump"))
        {
            gravity = -gravity;
        }

        rb.AddForce(gravity);
    }

    private void FixedUpdate()
    {

    }
}