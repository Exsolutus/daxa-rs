
use proc_macro::TokenStream;
use quote::quote;
use syn;

use bitfield::*;



pub fn impl_slot(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    
    validate_fields(&ast.data);

    // Implement Slot trait
    let gen = quote! {
        impl Slot for #name {
            #[inline]
            fn is_zombie(&self) -> bool {
                self.zombie
            }
        }
    };
    gen.into()
}

fn validate_fields(data: &syn::Data) {
    match data {
        syn::Data::Struct(ref data) => {
            match data.fields {
                syn::Fields::Named(ref fields) => {
                    let zombie_field = fields.named.iter().find(|&field| {
                        field.ident.as_ref().unwrap() == "zombie" && is_type(&field.ty, "bool")
                    });

                    assert!(zombie_field.is_some(), "Structs deriving Slot should have a bool field named 'zombie'");
                },
                _ => ()
            }
        },
        _ => unimplemented!("Types deriving ResourceId should be structs.")
    };
}

fn is_type(ty: &syn::Type, target: &str) -> bool {
    if let syn::Type::Path(ref p) = ty {
        return p.path.segments.len() == 1 && p.path.segments[0].ident == target;
    } else {
        false
    }
}