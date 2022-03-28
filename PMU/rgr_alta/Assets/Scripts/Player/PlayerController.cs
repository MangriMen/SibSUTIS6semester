using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [SerializeField]
    private Vector2 gravity = new Vector2(-9.81f, 0f);

    [SerializeField]
    private GameObject screenOfDeadPrefab;
    [SerializeField]
    private Canvas cnv;

    private GameObject screenOfDead;

    private Rigidbody2D rb;

    private bool isGrounded = false;

    public Transform left;
    public Transform right;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    void Update()
    {
        if (isGrounded && Input.GetButtonDown("Jump"))
        {
            gravity = -gravity;
        }

        rb.AddForce(gravity);
    }

    private void FixedUpdate()
    {
        isGrounded = false;

        foreach(Collider2D col in getLeftRightColliders())
        {
            if (col.gameObject.layer == 6)
            {
                isGrounded = true;
                break;
            }
            else if (col.gameObject.CompareTag("KillBox"))
            {
                Destroy(gameObject, 0.8f);
                if (screenOfDead == null)
                {
                    screenOfDead = Instantiate(screenOfDeadPrefab, cnv.transform);
                }
            }
        }
    }

    private List<Collider2D> getLeftRightColliders()
    {
        List<Collider2D> all = new List<Collider2D>();
        all.AddRange(Physics2D.OverlapCircleAll(left.transform.position, 0.05f));
        all.AddRange(Physics2D.OverlapCircleAll(right.transform.position, 0.05f));
        return all;
    }
}